%%  Comprehensive test for all ScryNeuro FFI functions
:- use_module(library(ffi)).
:- use_module(library(iso_ext)).

expect(Label, Goal) :-
    ( call(Goal) -> true
    ; throw(error(test_failed(Label), test_comprehensive/0))
    ).

lib_path(Path) :-
    ( catch((open('./libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.dylib"
    ; catch((open('./libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "./libscryneuro.so"
    ; catch((open('../libscryneuro.dylib', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.dylib"
    ; catch((open('../libscryneuro.so', read, S), close(S)), _, fail) ->
        Path = "../libscryneuro.so"
    ; throw(error("Could not find libscryneuro.dylib or libscryneuro.so", lib_path/1))
    ).


 init :-
    lib_path(LibPath),
    ( use_foreign_module(LibPath, [
        'spy_init'([], sint32),
        'spy_finalize'([], void),
        'spy_last_error'([], cstr),
        'spy_last_error_clear'([], void),
        'spy_handle_count'([], sint64),
        'spy_drop'([ptr], void),
        'spy_eval'([cstr], ptr),
        'spy_exec'([cstr], sint32),
        'spy_import'([cstr], ptr),
        'spy_getattr'([ptr, cstr], ptr),
        'spy_setattr'([ptr, cstr, ptr], sint32),
        'spy_call0'([ptr], ptr),
        'spy_call1'([ptr, ptr], ptr),
        'spy_call2'([ptr, ptr, ptr], ptr),
        'spy_call3'([ptr, ptr, ptr, ptr], ptr),
        'spy_calln'([ptr, ptr], ptr),
        'spy_invoke0'([ptr, cstr], ptr),
        'spy_invoke1'([ptr, cstr, ptr], ptr),
        'spy_invoke2'([ptr, cstr, ptr, ptr], ptr),
        'spy_invoken'([ptr, cstr, ptr], ptr),
        'spy_to_str'([ptr], cstr),
        'spy_to_repr'([ptr], cstr),
        'spy_to_int'([ptr], sint64),
        'spy_to_float'([ptr], f64),
        'spy_to_bool'([ptr], sint32),
        'spy_from_int'([sint64], ptr),
        'spy_from_float'([f64], ptr),
        'spy_from_bool'([sint32], ptr),
        'spy_from_str'([cstr], ptr),
        'spy_none'([], ptr),
        'spy_is_none'([ptr], sint32),
        'spy_to_json'([ptr], cstr),
        'spy_from_json'([cstr], ptr),
        'spy_list_new'([], ptr),
        'spy_list_append'([ptr, ptr], sint32),
        'spy_list_get'([ptr, sint64], ptr),
        'spy_list_len'([ptr], sint64),
        'spy_dict_new'([], ptr),
        'spy_dict_set'([ptr, cstr, ptr], sint32),
        'spy_dict_get'([ptr, cstr], ptr)
    ]) -> true ; throw(error("Failed to load foreign module (check DYLD_LIBRARY_PATH on macOS or LD_LIBRARY_PATH on Linux)", init/0))).

test :-
    write('1. spy_init: '),
    ffi:'spy_init'(S), write(S), nl,
    expect('1. spy_init', S =:= 0),
    setup_call_cleanup(true, test_body, ffi:'spy_finalize').

test_body :-

    write('2. from_int(42)->to_int: '),
    ffi:'spy_from_int'(42, HI),
    ffi:'spy_to_int'(HI, VI),
    write(VI), nl,
    expect('2. from_int', VI =:= 42),
    ffi:'spy_drop'(HI),

    write('3. from_float(3.14)->to_float: '),
    ffi:'spy_from_float'(3.14, HF),
    ffi:'spy_to_float'(HF, VF),
    write(VF), nl,
    expect('3. from_float', (VF > 3.139, VF < 3.141)),
    ffi:'spy_drop'(HF),

    write('4. from_bool(1)->to_bool: '),
    ffi:'spy_from_bool'(1, HB),
    ffi:'spy_to_bool'(HB, VB),
    write(VB), nl,
    expect('4. from_bool', VB =:= 1),
    ffi:'spy_drop'(HB),

    write('5. from_str(test)->to_str: '),
    ffi:'spy_from_str'("test", HS),
    ffi:'spy_to_str'(HS, VS),
    write(VS), nl,
    expect('5. from_str', VS = "test"),
    ffi:'spy_drop'(HS),

    write('6. eval(2**10): '),
    ffi:'spy_eval'("2**10", HE),
    ffi:'spy_to_int'(HE, VE),
    write(VE), nl,
    expect('6. eval', VE =:= 1024),
    ffi:'spy_drop'(HE),

    write('7. exec/eval readback: '),
    ffi:'spy_exec'("_tv = 99", SE),
    write('exec='), write(SE), write(' '),
    ffi:'spy_eval'("_tv", HEV),
    ffi:'spy_to_int'(HEV, VEV),
    write('val='), write(VEV), nl,
    expect('7. exec/eval readback', (SE =:= 0, VEV =:= 99)),
    ffi:'spy_drop'(HEV),

    write('8. import math: '),
    ffi:'spy_import'("math", HM),
    write(HM), nl,
    expect('8. import math', HM =\= 0),

    write('9. math.pi: '),
    ffi:'spy_getattr'(HM, "pi", HPI),
    ffi:'spy_to_float'(HPI, VPI),
    write(VPI), nl,
    expect('9. math.pi', (VPI > 3.14, VPI < 3.15)),
    ffi:'spy_drop'(HPI),

    write('10. math.sqrt(16): '),
    ffi:'spy_getattr'(HM, "sqrt", HSqrt),
    ffi:'spy_from_float'(16.0, H16),
    ffi:'spy_call1'(HSqrt, H16, HRes10),
    ffi:'spy_to_float'(HRes10, VR10),
    write(VR10), nl,
    expect('10. math.sqrt', VR10 =:= 4.0),
    ffi:'spy_drop'(HSqrt), ffi:'spy_drop'(H16), ffi:'spy_drop'(HRes10),

    write('11. math.factorial(5): '),
    ffi:'spy_from_int'(5, H5),
    ffi:'spy_invoke1'(HM, "factorial", H5, HFact),
    ffi:'spy_to_int'(HFact, VFact),
    write(VFact), nl,
    expect('11. math.factorial', VFact =:= 120),
    ffi:'spy_drop'(H5), ffi:'spy_drop'(HFact), ffi:'spy_drop'(HM),

    write('12. none/is_none: '),
    ffi:'spy_none'(HN),
    ffi:'spy_is_none'(HN, IN),
    write(IN), nl,
    expect('12. none/is_none', IN =:= 1),
    ffi:'spy_drop'(HN),

    write('13. repr(42): '),
    ffi:'spy_from_int'(42, HR13),
    ffi:'spy_to_repr'(HR13, VR13),
    write(VR13), nl,
    expect('13. repr', VR13 = "42"),
    ffi:'spy_drop'(HR13),

    write('14. list: '),
    ffi:'spy_list_new'(HL),
    ffi:'spy_from_int'(10, H10L),
    ffi:'spy_from_int'(20, H20L),
    ffi:'spy_list_append'(HL, H10L, _),
    ffi:'spy_list_append'(HL, H20L, _),
    ffi:'spy_list_len'(HL, LLen),
    write('len='), write(LLen), write(' '),
    ffi:'spy_list_get'(HL, 0, HLG),
    ffi:'spy_to_int'(HLG, VLG),
    write('get(0)='), write(VLG), nl,
    expect('14. list', (LLen =:= 2, VLG =:= 10)),
    ffi:'spy_drop'(HLG), ffi:'spy_drop'(H10L), ffi:'spy_drop'(H20L), ffi:'spy_drop'(HL),

    write('15. dict: '),
    ffi:'spy_dict_new'(HD),
    ffi:'spy_from_str'("world", HDV),
    ffi:'spy_dict_set'(HD, "hello", HDV, _),
    ffi:'spy_dict_get'(HD, "hello", HDG),
    ffi:'spy_to_str'(HDG, VDG),
    write(VDG), nl,
    expect('15. dict', VDG = "world"),
    ffi:'spy_drop'(HDG), ffi:'spy_drop'(HDV), ffi:'spy_drop'(HD),

    write('16. JSON: '),
    ffi:'spy_from_json'("[1,2,3]", HJ),
    ffi:'spy_to_json'(HJ, VJ),
    write(VJ), nl,
    expect('16. JSON', VJ = "[1, 2, 3]"),
    ffi:'spy_drop'(HJ),

    write('17. call0 list(): '),
    ffi:'spy_eval'("list", HListCls),
    ffi:'spy_call0'(HListCls, HEmpty),
    ffi:'spy_list_len'(HEmpty, ELen),
    write('len='), write(ELen), nl,
    expect('17. call0', ELen =:= 0),
    ffi:'spy_drop'(HEmpty), ffi:'spy_drop'(HListCls),

    write('18. pow(2,10): '),
    ffi:'spy_eval'("pow", HPow),
    ffi:'spy_from_int'(2, HC2),
    ffi:'spy_from_int'(10, HC10),
    ffi:'spy_call2'(HPow, HC2, HC10, HCR),
    ffi:'spy_to_int'(HCR, VCR),
    write(VCR), nl,
    expect('18. pow', VCR =:= 1024),
    ffi:'spy_drop'(HPow), ffi:'spy_drop'(HC2), ffi:'spy_drop'(HC10), ffi:'spy_drop'(HCR),

    write('19. lambda(1,2,3): '),
    ffi:'spy_eval'("lambda a,b,c: a+b+c", HLam),
    ffi:'spy_from_int'(1, HCA),
    ffi:'spy_from_int'(2, HCB),
    ffi:'spy_from_int'(3, HCC),
    ffi:'spy_call3'(HLam, HCA, HCB, HCC, HCR3),
    ffi:'spy_to_int'(HCR3, VCR3),
    write(VCR3), nl,
    expect('19. call3', VCR3 =:= 6),
    ffi:'spy_drop'(HLam), ffi:'spy_drop'(HCA), ffi:'spy_drop'(HCB), ffi:'spy_drop'(HCC), ffi:'spy_drop'(HCR3),

    write('20. invoke0 upper: '),
    ffi:'spy_from_str'("hello", HSI0),
    ffi:'spy_invoke0'(HSI0, "upper", HU0),
    ffi:'spy_to_str'(HU0, VU0),
    write(VU0), nl,
    expect('20. invoke0', VU0 = "HELLO"),
    ffi:'spy_drop'(HSI0), ffi:'spy_drop'(HU0),

    write('21. invoke2 replace: '),
    ffi:'spy_from_str'("hello world", HSI2),
    ffi:'spy_from_str'("world", HSA1),
    ffi:'spy_from_str'("prolog", HSA2),
    ffi:'spy_invoke2'(HSI2, "replace", HSA1, HSA2, HRI2),
    ffi:'spy_to_str'(HRI2, VRI2),
    write(VRI2), nl,
    expect('21. invoke2', VRI2 = "hello prolog"),
    ffi:'spy_drop'(HSI2), ffi:'spy_drop'(HSA1), ffi:'spy_drop'(HSA2), ffi:'spy_drop'(HRI2),

    write('22. setattr/getattr: '),
    ffi:'spy_exec'("class _O: pass", _),
    ffi:'spy_eval'("_O()", HOb),
    ffi:'spy_from_int'(99, HOV),
    ffi:'spy_setattr'(HOb, "x", HOV, SAR),
    write('set='), write(SAR), write(' '),
    ffi:'spy_getattr'(HOb, "x", HOX),
    ffi:'spy_to_int'(HOX, VOX),
    write('get='), write(VOX), nl,
    expect('22. setattr/getattr', (SAR =:= 0, VOX =:= 99)),
    ffi:'spy_drop'(HOV), ffi:'spy_drop'(HOX), ffi:'spy_drop'(HOb),

    write('23. calln sum4: '),
    ffi:'spy_eval'("lambda a,b,c,d: a+b+c+d", HSum4),
    ffi:'spy_list_new'(HArgs),
    ffi:'spy_from_int'(1, HA1),
    ffi:'spy_from_int'(2, HA2),
    ffi:'spy_from_int'(3, HA3),
    ffi:'spy_from_int'(4, HA4),
    ffi:'spy_list_append'(HArgs, HA1, _),
    ffi:'spy_list_append'(HArgs, HA2, _),
    ffi:'spy_list_append'(HArgs, HA3, _),
    ffi:'spy_list_append'(HArgs, HA4, _),
    ffi:'spy_calln'(HSum4, HArgs, HSumRes),
    ffi:'spy_to_int'(HSumRes, VSumRes),
    write(VSumRes), nl,
    expect('23. calln', VSumRes =:= 10),
    ffi:'spy_drop'(HSumRes), ffi:'spy_drop'(HA4), ffi:'spy_drop'(HA3), ffi:'spy_drop'(HA2), ffi:'spy_drop'(HA1),
    ffi:'spy_drop'(HArgs), ffi:'spy_drop'(HSum4),

    write('24. invoken replace: '),
    ffi:'spy_from_str'("hello world", HInvStr),
    ffi:'spy_list_new'(HInvArgs),
    ffi:'spy_from_str'("world", HInvA1),
    ffi:'spy_from_str'("prolog", HInvA2),
    ffi:'spy_list_append'(HInvArgs, HInvA1, _),
    ffi:'spy_list_append'(HInvArgs, HInvA2, _),
    ffi:'spy_invoken'(HInvStr, "replace", HInvArgs, HInvRes),
    ffi:'spy_to_str'(HInvRes, VInvRes),
    write(VInvRes), nl,
    expect('24. invoken', VInvRes = "hello prolog"),
    ffi:'spy_drop'(HInvRes), ffi:'spy_drop'(HInvA2), ffi:'spy_drop'(HInvA1), ffi:'spy_drop'(HInvArgs),
    ffi:'spy_drop'(HInvStr),

    write('25. error(1/0): '),
    ffi:'spy_eval'("1/0", HErr),
    write('h='), write(HErr), write(' '),
    ffi:'spy_last_error'(EMsg),
    write(EMsg), nl,
    expect('25. error', (HErr =:= 0, EMsg \= [])),
    ffi:'spy_last_error_clear',

    write('26. handle_count: '),
    ffi:'spy_handle_count'(FC),
    write(FC), nl,
    expect('26. handle_count', FC =:= 0),
    write('=== ALL 26 TESTS PASSED ==='), nl.

run_tests :-
    ( catch((init, test), Error,
            ( write('Comprehensive tests failed: '), writeq(Error), nl, halt(1) )) ->
        halt(0)
    ; write('Comprehensive tests failed without an exception.'), nl, halt(1)
    ).

:- initialization(run_tests).
