%% ===========================================================================
%% ScryNeuro Tool Predicates (Prolog wrappers for Python tools)
%% ===========================================================================

:- module(scryer_tool_predicates, [
    tool_list_available/1,
    tool_shell_exec/4,
    tool_shell_exec/5,
    tool_web_fetch/4,
    tool_web_fetch/5,
    tool_read_file/2,
    tool_read_file/3,
    tool_write_file/3,
    tool_write_file/4,
    tool_list_dir/2,
    tool_list_dir/3,
    tool_grep_text/3,
    tool_grep_text/4,
    tool_call_json/3
]).

:- use_module('scryer_py').
:- use_module(library(lists)).

tool_call_json(ToolName, Options, ResultJson) :-
    py_import("scryer_agent.tool_runtime", Runtime),
    atom_chars(ToolName, ToolChars),
    py_from_str(ToolChars, PyToolName),
    py_dict_new(Kwargs),
    load_options(Kwargs, Options),
    py_call(Runtime, "tool_call", PyToolName, Kwargs, ResultH),
    py_to_json(ResultH, ResultJson),
    py_free(ResultH),
    py_free(Kwargs),
    py_free(PyToolName),
    py_free(Runtime).

tool_list_available(ToolsJson) :-
    py_import("scryer_agent.tool_runtime", Runtime),
    py_call(Runtime, "tool_list_available", ToolsH),
    py_to_json(ToolsH, ToolsJson),
    py_free(ToolsH),
    py_free(Runtime).

tool_shell_exec(Command, Code, Stdout, Stderr) :-
    tool_shell_exec(Command, [], Code, Stdout, Stderr).

tool_shell_exec(Command, Options, Code, Stdout, Stderr) :-
    append([command=Command], Options, FullOptions),
    tool_call_json(shell_exec, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "returncode", CodeH),
    py_to_int(CodeH, Code),
    py_dict_get(Result, "stdout", OutH),
    py_to_str(OutH, Stdout),
    py_dict_get(Result, "stderr", ErrH),
    py_to_str(ErrH, Stderr),
    py_free(ErrH),
    py_free(OutH),
    py_free(CodeH),
    py_free(Result),
    py_free(Obj).

tool_web_fetch(Url, StatusCode, ContentType, Preview) :-
    tool_web_fetch(Url, [], StatusCode, ContentType, Preview).

tool_web_fetch(Url, Options, StatusCode, ContentType, Preview) :-
    append([url=Url], Options, FullOptions),
    tool_call_json(web_fetch, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "status_code", StatusH),
    py_to_int(StatusH, StatusCode),
    py_dict_get(Result, "content_type", CTypeH),
    py_to_str(CTypeH, ContentType),
    py_dict_get(Result, "content_preview", PrevH),
    py_to_str(PrevH, Preview),
    py_free(PrevH),
    py_free(CTypeH),
    py_free(StatusH),
    py_free(Result),
    py_free(Obj).

tool_read_file(Path, Content) :-
    tool_read_file(Path, [], Content).

tool_read_file(Path, Options, Content) :-
    append([path=Path], Options, FullOptions),
    tool_call_json(read_file, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "content", ContentH),
    py_to_str(ContentH, Content),
    py_free(ContentH),
    py_free(Result),
    py_free(Obj).

tool_write_file(Path, Content, BytesWritten) :-
    tool_write_file(Path, Content, [], BytesWritten).

tool_write_file(Path, Content, Options, BytesWritten) :-
    append([path=Path, content=Content], Options, FullOptions),
    tool_call_json(write_file, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "bytes_written", BytesH),
    py_to_int(BytesH, BytesWritten),
    py_free(BytesH),
    py_free(Result),
    py_free(Obj).

tool_list_dir(Path, EntriesJson) :-
    tool_list_dir(Path, [], EntriesJson).

tool_list_dir(Path, Options, EntriesJson) :-
    append([path=Path], Options, FullOptions),
    tool_call_json(list_dir, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "entries", EntriesH),
    py_to_json(EntriesH, EntriesJson),
    py_free(EntriesH),
    py_free(Result),
    py_free(Obj).

tool_grep_text(Path, Pattern, MatchesJson) :-
    tool_grep_text(Path, Pattern, [], MatchesJson).

tool_grep_text(Path, Pattern, Options, MatchesJson) :-
    append([path=Path, pattern=Pattern], Options, FullOptions),
    tool_call_json(grep_text, FullOptions, Json),
    py_from_json(Json, Obj),
    py_dict_get(Obj, "result", Result),
    py_dict_get(Result, "matches", MatchesH),
    py_to_json(MatchesH, MatchesJson),
    py_free(MatchesH),
    py_free(Result),
    py_free(Obj).
