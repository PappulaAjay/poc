from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from poc.chat import GenConfig, generate, load_base, load_finetuned
from poc.data import build_qa_from_text_sources, extract_texts_from_uploads
from poc.training import TrainConfig, train_lora


APP_TITLE = "Healthcare SLM Fine-Tuning POC (CPU LoRA)"


def _init_state() -> None:
    st.session_state.setdefault("items", None)
    st.session_state.setdefault("examples", None)
    st.session_state.setdefault("last_run_dir", None)
    st.session_state.setdefault("live_metrics", [])


def _now_run_dir() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path("runs") / ts)


@st.cache_resource(show_spinner=False)
def _load_base_cached(model_id: str):
    return load_base(model_id)


def _metrics_df(metrics: list[dict]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    df = pd.DataFrame(metrics)
    df = df.sort_values("step")
    return df


def _plot_matplotlib(df: pd.DataFrame, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    d = df[df["event"] == "train"].copy()
    if y_col not in d.columns or d.empty:
        ax.text(0.1, 0.5, f"No data for {y_col}", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ax.plot(d["step"], d[y_col])
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    return fig


def main() -> None:
    _init_state()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.subheader("Model + training settings")
        model_id = st.text_input(
            "Base model (HF id)",
            value="HuggingFaceTB/SmolLM2-135M-Instruct",
            help="Small default so CPU LoRA finishes quickly. You can swap to a larger model if your CPU/RAM allow.",
        )
        epochs = st.slider("Epochs", min_value=1, max_value=3, value=2)
        max_steps = st.slider("Max steps (cap runtime)", min_value=20, max_value=400, value=160, step=10)
        max_seq_len = st.select_slider("Max sequence length", options=[128, 192, 256, 320, 384], value=256)

        st.divider()
        st.subheader("LoRA settings")
        lora_r = st.select_slider("LoRA rank (r)", options=[4, 8, 16, 32], value=8)
        lr = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)

        st.divider()
        st.subheader("CPU")
        num_threads = st.number_input(
            "torch num_threads (0 = default)",
            min_value=0,
            max_value=128,
            value=0,
            step=1,
        )

        st.divider()
        st.subheader("Dataset slice")
        max_examples = st.slider("Max training examples", min_value=40, max_value=400, value=200, step=20)

    tab_upload, tab_train, tab_chat = st.tabs(["1) Upload Data", "2–3) Train + Metrics", "4) Chat: Before vs After"])

    with tab_upload:
        st.subheader("Step 1: Upload healthcare data (PDF / CSV / TXT)")
        st.caption("Tip: If you upload a CSV with columns `question,answer`, the POC learns much better (still LoRA fine-tuning, not RAG).")
        uploads = st.file_uploader(
            "Upload files (max ~100MB total)",
            type=["pdf", "csv", "txt", "md", "log"],
            accept_multiple_files=True,
        )
        if uploads:
            items = extract_texts_from_uploads(uploads)
            st.session_state["items"] = items

            st.write("### Extracted sources")
            rows = []
            for it in items:
                if it["kind"] == "text":
                    rows.append(
                        {
                            "source": it["source"],
                            "kind": it["kind"],
                            "chars": len(it["text"] or ""),
                            "preview": (it["text"] or "")[:240].replace("\n", " "),
                        }
                    )
                else:
                    rows.append(
                        {
                            "source": it["source"],
                            "kind": it["kind"],
                            "chars": "",
                            "preview": f"{len(it['qa'])} Q&A rows",
                        }
                    )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            examples = build_qa_from_text_sources(items, max_examples=max_examples)
            st.session_state["examples"] = examples

            st.write("### Training examples (preview)")
            st.write(f"Built **{len(examples)}** supervised examples.")
            if examples:
                ex0 = examples[0]
                st.code(
                    "### Instruction:\n"
                    f"{ex0.instruction}\n\n"
                    "### User:\n"
                    f"{ex0.input}\n\n"
                    "### Assistant:\n"
                    f"{ex0.output[:800]}{'...' if len(ex0.output) > 800 else ''}"
                )

    with tab_train:
        st.subheader("Steps 2–3: CPU LoRA training + live metrics")
        examples = st.session_state.get("examples") or []
        if not examples:
            st.info("Upload data in the first tab to build a training dataset.")
        else:
            cols = st.columns([1, 1, 1])
            with cols[0]:
                train_n = st.number_input("Train examples", min_value=16, max_value=len(examples), value=min(160, len(examples)))
            with cols[1]:
                eval_n = st.number_input(
                    "Eval examples",
                    min_value=8,
                    max_value=max(8, len(examples) - int(train_n)),
                    value=min(40, max(8, len(examples) - int(train_n))),
                )
            with cols[2]:
                batch_size = st.selectbox("Batch size (CPU)", options=[1, 2, 4], index=1)

            grad_accum = st.selectbox("Grad accumulation", options=[1, 2, 4, 8, 16], index=3)
            eval_every_steps = st.selectbox("Eval every N steps", options=[25, 50, 75, 100], index=1)

            st.caption("Metrics shown: loss curve, perplexity (exp(loss)), token-level accuracy on supervised tokens.")
            start = st.button("Train (CPU LoRA)", type="primary")

            # live placeholders
            prog = st.progress(0)
            status = st.empty()
            c1, c2, c3 = st.columns(3)
            loss_ph = c1.empty()
            ppl_ph = c2.empty()
            acc_ph = c3.empty()
            fig1_ph, fig2_ph, fig3_ph = st.columns(3)
            loss_fig = fig1_ph.empty()
            ppl_fig = fig2_ph.empty()
            acc_fig = fig3_ph.empty()
            metrics_tbl = st.empty()

            if start:
                st.session_state["live_metrics"] = []
                run_dir = _now_run_dir()
                Path(run_dir).mkdir(parents=True, exist_ok=True)

                # callback for live updates
                def on_metric(m: dict) -> None:
                    st.session_state["live_metrics"].append(m)

                cfg = TrainConfig(
                    model_id=model_id,
                    run_dir=run_dir,
                    epochs=int(epochs),
                    max_steps=int(max_steps),
                    max_seq_len=int(max_seq_len),
                    train_examples=int(train_n),
                    eval_examples=int(eval_n),
                    lr=float(lr),
                    batch_size=int(batch_size),
                    grad_accum=int(grad_accum),
                    lora_r=int(lora_r),
                    eval_every_steps=int(eval_every_steps),
                    num_threads=int(num_threads),
                )

                t_start = time.time()
                total = int(max_steps)

                def refresh_ui():
                    df = _metrics_df(st.session_state["live_metrics"])
                    if df.empty:
                        return

                    # latest values
                    last_train = df[df["event"] == "train"].tail(1)
                    last_eval = df[df["event"] == "eval"].tail(1)
                    if not last_train.empty:
                        tl = float(last_train["train_loss"].iloc[0])
                        tp = float(last_train["train_ppl"].iloc[0])
                        ta = last_train.get("train_token_acc")
                        status.write(
                            f"Step **{int(last_train['step'].iloc[0])}** | "
                            f"train loss **{tl:.4f}** | ppl **{tp:.2f}**"
                            + (f" | token acc **{float(ta.iloc[0]):.3f}**" if ta is not None and not pd.isna(ta.iloc[0]) else "")
                        )
                        loss_ph.metric("Train loss", f"{tl:.4f}")
                        ppl_ph.metric("Train perplexity", f"{tp:.2f}")
                        if ta is not None and not pd.isna(ta.iloc[0]):
                            acc_ph.metric("Train token acc (approx)", f"{float(ta.iloc[0]):.3f}")
                        prog.progress(min(1.0, int(last_train["step"].iloc[0]) / max(1, total)))

                    if not last_eval.empty:
                        el = float(last_eval["eval_loss"].iloc[0])
                        ep = float(last_eval["eval_ppl"].iloc[0])
                        ea = float(last_eval["eval_token_acc"].iloc[0])
                        c1.caption(f"Latest eval loss: **{el:.4f}**")
                        c2.caption(f"Latest eval ppl: **{ep:.2f}**")
                        c3.caption(f"Latest eval token acc: **{ea:.3f}**")

                    # plots
                    loss_fig.pyplot(_plot_matplotlib(df, "train_loss", "Loss (train)"))
                    ppl_fig.pyplot(_plot_matplotlib(df, "train_ppl", "Perplexity (train)"))
                    acc_fig.pyplot(_plot_matplotlib(df, "train_token_acc", "Token accuracy (train, approx)"))

                    metrics_tbl.dataframe(df.tail(200), use_container_width=True, height=260)

                # run training (blocking but live-updating)
                result = train_lora(cfg, examples, ui_callback=lambda m: (on_metric(m), refresh_ui()))
                st.session_state["last_run_dir"] = result["run_dir"]
                seconds = round(time.time() - t_start, 2)

                st.success(
                    f"Training finished in **{seconds}s** | steps **{result['steps']}** | "
                    f"final eval loss **{result['final_eval']['eval_loss']:.4f}** | "
                    f"eval ppl **{result['final_eval']['eval_ppl']:.2f}** | "
                    f"eval token acc **{result['final_eval']['eval_token_acc']:.3f}**"
                )

            # If a previous run exists, let user load metrics
            last_run_dir = st.session_state.get("last_run_dir")
            if last_run_dir:
                st.write("### Last run artifacts")
                st.code(last_run_dir)
                metrics_file = Path(last_run_dir) / "metrics.jsonl"
                if metrics_file.exists():
                    with metrics_file.open("r", encoding="utf-8") as f:
                        ms = [json.loads(line) for line in f if line.strip()]
                    df = _metrics_df(ms)
                    st.dataframe(df.tail(200), use_container_width=True)

    with tab_chat:
        st.subheader("Step 4: Chatbot demo — Base vs Fine-tuned")
        st.caption("This compares the same prompt on the base model and the LoRA-adapted model.")

        last_run_dir = st.session_state.get("last_run_dir")
        if not last_run_dir:
            st.info("Train a model in the previous tab to enable the fine-tuned comparison.")

        user_prompt = st.text_area(
            "Ask a healthcare/policy question",
            value="Based on the uploaded dataset, does the policy cover Procedure 99213? What conditions or exclusions apply?",
            height=120,
        )

        gen_cols = st.columns(3)
        with gen_cols[0]:
            max_new = st.slider("Max new tokens", 64, 512, 220, 16)
        with gen_cols[1]:
            temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        with gen_cols[2]:
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

        gen_cfg = GenConfig(max_new_tokens=int(max_new), temperature=float(temp), top_p=float(top_p))

        go = st.button("Run comparison", type="primary")
        if go:
            with st.spinner("Loading base model (CPU)..."):
                tok_b, model_b = _load_base_cached(model_id)
            with st.spinner("Generating (base)..."):
                base_ans = generate(tok_b, model_b, user_prompt, gen_cfg)

            fin_ans = None
            if last_run_dir:
                adapter_dir = str(Path(last_run_dir) / "lora_adapter")
                with st.spinner("Loading fine-tuned adapter (CPU)..."):
                    tok_f, model_f = load_finetuned(model_id, adapter_dir)
                with st.spinner("Generating (fine-tuned)..."):
                    fin_ans = generate(tok_f, model_f, user_prompt, gen_cfg)

            left, right = st.columns(2)
            with left:
                st.write("### Before training (base)")
                st.text_area("Base model answer", value=base_ans, height=260)
            with right:
                st.write("### After training (fine-tuned LoRA)")
                st.text_area("Fine-tuned answer", value=fin_ans or "Train first to see this.", height=260)


if __name__ == "__main__":
    main()

