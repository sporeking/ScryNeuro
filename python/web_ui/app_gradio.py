from __future__ import annotations

import importlib
from typing import Any

_GRADIO_IMPORT_ERROR: Exception | None = None

from agent_adapter import (
    close_agent,
    create_or_replace_agent,
    get_agent_messages,
    get_agent_summary,
    get_profile,
    list_agents,
    list_models,
    list_profiles,
    list_providers,
    list_skills,
    list_tools,
    parse_json_object,
    reset_agent,
    run_agent,
    to_chatbot_messages,
    to_json_text,
    trace_agent,
)


def _safe_call(fn, *args, **kwargs) -> tuple[str, Any]:
    try:
        result = fn(*args, **kwargs)
        return "ok", result
    except Exception as e:
        return "error", {"error": str(e)}


def _get_gradio_module() -> Any:
    global _GRADIO_IMPORT_ERROR
    try:
        return importlib.import_module("gradio")
    except ModuleNotFoundError as e:
        _GRADIO_IMPORT_ERROR = e
        return None


def _refresh_catalogs() -> tuple[list[str], list[str], list[str], list[str], str]:
    providers = list_providers()
    profiles = list_profiles()
    tool_names = [str(x.get("name", "")) for x in list_tools() if x.get("name")]
    skill_names = [str(x.get("name", "")) for x in list_skills() if x.get("name")]
    info = {
        "providers": providers,
        "profiles": profiles,
        "tools": tool_names,
        "skills": skill_names,
        "active_agents": list_agents(),
    }
    return providers, profiles, tool_names, skill_names, to_json_text(info)


def on_profile_changed(profile_name: str) -> tuple[str, str, str, str]:
    if not profile_name:
        return "", "", "", "{}"
    status, data = _safe_call(get_profile, profile_name)
    if status == "error":
        return "", "", "", to_json_text(data)
    provider = str(data.get("provider", ""))
    model = str(data.get("model", ""))
    base_url = str(data.get("base_url", ""))
    profile_json = to_json_text(data)
    return provider, model, base_url, profile_json


def on_provider_changed(provider: str) -> tuple[str, str]:
    models = list_models(provider)
    model_text = ", ".join(models)
    return model_text, to_json_text({"provider": provider, "models": models})


def on_create_agent(
    agent_name: str,
    profile_name: str,
    override_json_text: str,
    selected_tools: list[str],
    selected_skills: list[str],
) -> tuple[str, str, list[dict[str, str]], str]:
    status, overrides = _safe_call(parse_json_object, override_json_text)
    if status == "error":
        return status.upper(), to_json_text({"result": overrides}), [], "{}"
    status, data = _safe_call(
        create_or_replace_agent,
        name=agent_name,
        profile_name=profile_name,
        overrides=overrides,
        tool_names=selected_tools,
        skill_names=selected_skills,
    )
    chat: list[dict[str, str]] = []
    summary_text = "{}"
    if status == "ok":
        s_status, s_data = _safe_call(get_agent_summary, agent_name)
        if s_status == "ok":
            summary_text = to_json_text(s_data)
    active = list_agents()
    payload = {"active_agents": active, "result": data}
    return status.upper(), to_json_text(payload), chat, summary_text


def on_run_agent(
    agent_name: str,
    user_input: str,
    run_options_text: str,
) -> tuple[str, str, str, list[dict[str, str]], str]:
    status, options = _safe_call(parse_json_object, run_options_text)
    if status == "error":
        return status.upper(), "", to_json_text(options), [], "{}"
    status, data = _safe_call(
        run_agent,
        name=agent_name,
        user_input=user_input,
        run_options=options,
    )
    if status == "error":
        return status.upper(), "", to_json_text(data), [], "{}"
    response = str(data.get("response", ""))
    msg_status, msg_data = _safe_call(get_agent_messages, agent_name)
    if msg_status == "ok":
        chat_pairs = to_chatbot_messages(msg_data)
    else:
        chat_pairs = []
    summary_status, summary_data = _safe_call(get_agent_summary, agent_name)
    summary_text = to_json_text(
        summary_data if summary_status == "ok" else summary_data
    )
    return status.upper(), response, to_json_text(data), chat_pairs, summary_text


def on_trace_agent(agent_name: str) -> tuple[str, str]:
    status, data = _safe_call(trace_agent, agent_name)
    return status.upper(), to_json_text(data)


def on_refresh_conversation(agent_name: str) -> tuple[str, list[dict[str, str]], str]:
    status, data = _safe_call(get_agent_messages, agent_name)
    if status != "ok":
        return status.upper(), [], to_json_text(data)
    chat_pairs = to_chatbot_messages(data)
    s_status, s_data = _safe_call(get_agent_summary, agent_name)
    summary_text = to_json_text(s_data if s_status == "ok" else s_data)
    return status.upper(), chat_pairs, summary_text


def on_close_agent(agent_name: str) -> tuple[str, str, list[dict[str, str]], str]:
    status, data = _safe_call(close_agent, agent_name)
    payload = {
        "closed": data if status == "ok" else False,
        "active_agents": list_agents(),
        "detail": data,
    }
    return status.upper(), to_json_text(payload), [], "{}"


def on_reset_agent(agent_name: str) -> tuple[str, str, list[dict[str, str]], str]:
    status, data = _safe_call(reset_agent, agent_name)
    if status != "ok":
        return status.upper(), to_json_text(data), [], "{}"
    s_status, s_data = _safe_call(get_agent_summary, agent_name)
    summary_text = to_json_text(s_data if s_status == "ok" else s_data)
    payload = {"result": data, "active_agents": list_agents()}
    return status.upper(), to_json_text(payload), [], summary_text


def build_app() -> Any:
    gr = _get_gradio_module()
    if gr is None:
        raise RuntimeError(
            "Gradio is not installed. Install it with: pip install gradio "
            "(or use a virtualenv: python -m venv .venv && .venv/bin/pip install gradio)"
        ) from _GRADIO_IMPORT_ERROR

    providers, profiles, tool_names, skill_names, init_info = _refresh_catalogs()
    initial_profile = profiles[0] if profiles else ""
    initial_provider, initial_model, initial_base_url, initial_profile_json = (
        on_profile_changed(initial_profile)
    )

    with gr.Blocks(title="ScryNeuro Agent Playground") as demo:
        gr.Markdown("# ScryNeuro Agent Playground (Gradio)")
        gr.Markdown("用于快速测试：创建 Agent、启用工具/技能、运行任务、查看 trace。")

        with gr.Row():
            with gr.Column(scale=1):
                agent_name = gr.Textbox(
                    label="Agent Name",
                    value="ui_agent",
                    placeholder="例如: research_agent",
                )
                profile_name = gr.Dropdown(
                    label="Profile",
                    choices=profiles,
                    value=initial_profile if initial_profile else None,
                )
                provider = gr.Dropdown(
                    label="Provider (from profile)",
                    choices=providers,
                    interactive=False,
                    value=initial_provider or None,
                )
                model = gr.Textbox(
                    label="Model (from profile)",
                    interactive=False,
                    value=initial_model,
                )
                base_url = gr.Textbox(
                    label="Base URL (from profile)",
                    interactive=False,
                    value=initial_base_url,
                )

                selected_tools = gr.CheckboxGroup(
                    label="Enable Builtin Tools",
                    choices=tool_names,
                    value=["web_fetch", "shell_exec", "read_file", "write_file"],
                )
                selected_skills = gr.CheckboxGroup(
                    label="Load Skills",
                    choices=skill_names,
                    value=[],
                )

                override_json = gr.Code(
                    label="Create Overrides (JSON object)",
                    language="json",
                    value="{}",
                )
                create_btn = gr.Button("Create / Replace Agent", variant="primary")
                create_status = gr.Textbox(label="Create Status", interactive=False)
                create_out = gr.Code(
                    label="Create Output",
                    language="json",
                    interactive=False,
                    value=initial_profile_json,
                )

            with gr.Column(scale=1):
                task_input = gr.Textbox(
                    label="Task Input",
                    lines=6,
                    placeholder="输入任务，例如：抓取某网页并输出markdown摘要",
                )
                run_options = gr.Code(
                    label="Run Options (JSON object)",
                    language="json",
                    value='{"max_steps": 5, "max_auto_tools": 4, "temperature": 0.0, "max_tokens": 900}',
                )

                run_btn = gr.Button("Run", variant="primary")
                run_status = gr.Textbox(label="Run Status", interactive=False)
                run_response = gr.Textbox(
                    label="Agent Response", lines=6, interactive=False
                )
                run_out = gr.Code(
                    label="Run Output JSON", language="json", interactive=False
                )

                with gr.Row():
                    trace_btn = gr.Button("Get Trace")
                    refresh_conv_btn = gr.Button("Refresh Conversation")
                    reset_btn = gr.Button("Reset Conversation")
                    close_btn = gr.Button("Close Agent")
                trace_status = gr.Textbox(label="Trace/Close Status", interactive=False)
                trace_out = gr.Code(
                    label="Trace / Close Output", language="json", interactive=False
                )
                chat_view = gr.Chatbot(label="Conversation", height=360)
                conversation_summary = gr.Code(
                    label="Conversation Summary",
                    language="json",
                    value="{}",
                    interactive=False,
                )

        with gr.Accordion("Catalog & Debug Info", open=False):
            catalog_info = gr.Code(
                label="Catalog Info", language="json", value=init_info
            )
            refresh_btn = gr.Button("Refresh Catalog")

        profile_name.change(
            fn=on_profile_changed,
            inputs=[profile_name],
            outputs=[provider, model, base_url, create_out],
        )

        provider.change(
            fn=on_provider_changed,
            inputs=[provider],
            outputs=[model, catalog_info],
        )

        create_btn.click(
            fn=on_create_agent,
            inputs=[
                agent_name,
                profile_name,
                override_json,
                selected_tools,
                selected_skills,
            ],
            outputs=[create_status, create_out, chat_view, conversation_summary],
        )

        run_btn.click(
            fn=on_run_agent,
            inputs=[agent_name, task_input, run_options],
            outputs=[
                run_status,
                run_response,
                run_out,
                chat_view,
                conversation_summary,
            ],
        )

        trace_btn.click(
            fn=on_trace_agent,
            inputs=[agent_name],
            outputs=[trace_status, trace_out],
        )

        refresh_conv_btn.click(
            fn=on_refresh_conversation,
            inputs=[agent_name],
            outputs=[trace_status, chat_view, conversation_summary],
        )

        reset_btn.click(
            fn=on_reset_agent,
            inputs=[agent_name],
            outputs=[trace_status, trace_out, chat_view, conversation_summary],
        )

        close_btn.click(
            fn=on_close_agent,
            inputs=[agent_name],
            outputs=[trace_status, trace_out, chat_view, conversation_summary],
        )

        refresh_btn.click(
            fn=lambda: _refresh_catalogs()[4],
            inputs=[],
            outputs=[catalog_info],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
