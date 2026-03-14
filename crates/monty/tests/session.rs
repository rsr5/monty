//! Tests for `MontySession` — calling Python functions from Rust.

use monty::{MontyObject, MontyRun, MontySession, NoLimitTracker, PrintWriter};

/// Helper to create a session from Python code.
fn session(code: &str) -> MontySession<NoLimitTracker> {
    let runner = MontyRun::new(code.to_owned(), "session_test.py", vec![]).unwrap();
    runner.into_session(NoLimitTracker).unwrap()
}

// ===========================================================================
// Basic function calls
// ===========================================================================

#[test]
fn call_simple_function() {
    let mut s = session("def add(a, b): return a + b");
    let result = s
        .call_function("add", vec![MontyObject::Int(2), MontyObject::Int(3)])
        .unwrap();
    assert_eq!(result, MontyObject::Int(5));
}

#[test]
fn call_function_no_args() {
    let mut s = session("def greet(): return 'hello'");
    let result = s.call_function("greet", vec![]).unwrap();
    assert_eq!(result, MontyObject::String("hello".to_owned()));
}

#[test]
fn call_function_returns_none() {
    let mut s = session("def noop(): pass");
    let result = s.call_function("noop", vec![]).unwrap();
    assert_eq!(result, MontyObject::None);
}

#[test]
fn call_function_one_arg() {
    let mut s = session("def double(x): return x * 2");
    let result = s.call_function("double", vec![MontyObject::Int(21)]).unwrap();
    assert_eq!(result, MontyObject::Int(42));
}

#[test]
fn call_function_string_args() {
    let mut s = session("def concat(a, b): return a + b");
    let result = s
        .call_function(
            "concat",
            vec![
                MontyObject::String("hello ".to_owned()),
                MontyObject::String("world".to_owned()),
            ],
        )
        .unwrap();
    assert_eq!(result, MontyObject::String("hello world".to_owned()));
}

#[test]
fn call_function_multiple_times() {
    let mut s = session("def inc(x): return x + 1");
    for i in 0..5 {
        let result = s.call_function("inc", vec![MontyObject::Int(i)]).unwrap();
        assert_eq!(result, MontyObject::Int(i + 1));
    }
}

#[test]
fn call_function_with_list() {
    let mut s = session("def length(lst): return len(lst)");
    let result = s
        .call_function(
            "length",
            vec![MontyObject::List(vec![
                MontyObject::Int(1),
                MontyObject::Int(2),
                MontyObject::Int(3),
            ])],
        )
        .unwrap();
    assert_eq!(result, MontyObject::Int(3));
}

// ===========================================================================
// State persistence
// ===========================================================================

#[test]
fn session_retains_global_state() {
    let mut s = session(
        "\
counter = 0
def increment():
    global counter
    counter = counter + 1
    return counter
",
    );
    assert_eq!(s.call_function("increment", vec![]).unwrap(), MontyObject::Int(1));
    assert_eq!(s.call_function("increment", vec![]).unwrap(), MontyObject::Int(2));
    assert_eq!(s.call_function("increment", vec![]).unwrap(), MontyObject::Int(3));
}

#[test]
fn session_multiple_functions() {
    let mut s = session(
        "\
def add(a, b): return a + b
def mul(a, b): return a * b
",
    );
    assert_eq!(
        s.call_function("add", vec![MontyObject::Int(3), MontyObject::Int(4)])
            .unwrap(),
        MontyObject::Int(7)
    );
    assert_eq!(
        s.call_function("mul", vec![MontyObject::Int(3), MontyObject::Int(4)])
            .unwrap(),
        MontyObject::Int(12)
    );
}

#[test]
fn session_function_calls_other_function() {
    let mut s = session(
        "\
def double(x): return x * 2
def quadruple(x): return double(double(x))
",
    );
    let result = s.call_function("quadruple", vec![MontyObject::Int(5)]).unwrap();
    assert_eq!(result, MontyObject::Int(20));
}

// ===========================================================================
// Closures and defaults
// ===========================================================================

#[test]
fn call_function_with_defaults() {
    let mut s = session("def greet(name, greeting='Hello'): return greeting + ' ' + name");
    let result = s
        .call_function("greet", vec![MontyObject::String("world".to_owned())])
        .unwrap();
    assert_eq!(result, MontyObject::String("Hello world".to_owned()));
}

#[test]
fn call_closure() {
    let mut s = session(
        "\
def make_adder(n):
    def adder(x):
        return x + n
    return adder

add5 = make_adder(5)
",
    );
    let result = s.call_function("add5", vec![MontyObject::Int(10)]).unwrap();
    assert_eq!(result, MontyObject::Int(15));
}

// ===========================================================================
// Error handling
// ===========================================================================

#[test]
fn call_nonexistent_function() {
    let mut s = session("def foo(): return 1");
    let err = s.call_function("bar", vec![]).unwrap_err();
    assert!(err.to_string().contains("name 'bar' is not defined"), "got: {err}");
}

#[test]
fn call_non_callable() {
    let mut s = session("x = 42");
    let err = s.call_function("x", vec![]).unwrap_err();
    assert!(err.to_string().contains("not callable"), "got: {err}");
}

#[test]
fn call_function_raises_exception() {
    let mut s = session("def boom(): raise ValueError('kaboom')");
    let err = s.call_function("boom", vec![]).unwrap_err();
    assert!(err.to_string().contains("kaboom"), "got: {err}");
}

#[test]
fn call_function_wrong_arg_count() {
    let mut s = session("def add(a, b): return a + b");
    let err = s.call_function("add", vec![MontyObject::Int(1)]).unwrap_err();
    assert!(err.to_string().contains("argument"), "got: {err}");
}

// ===========================================================================
// Introspection
// ===========================================================================

#[test]
fn function_names() {
    let s = session(
        "\
x = 42
def foo(): pass
def bar(): pass
",
    );
    let mut names = s.function_names();
    names.sort_unstable();
    assert_eq!(names, vec!["bar", "foo"]);
}

#[test]
fn has_function() {
    let s = session("def my_func(): pass\nx = 10");
    assert!(s.has_function("my_func"));
    assert!(!s.has_function("x")); // not callable
    assert!(!s.has_function("nonexistent"));
}

// ===========================================================================
// Print capturing
// ===========================================================================

#[test]
fn call_function_captures_print() {
    let mut s = session("def say_hello(name): print('Hello ' + name)");
    let mut output = String::new();
    let result = s
        .call_function_with_print(
            "say_hello",
            vec![MontyObject::String("world".to_owned())],
            PrintWriter::Collect(&mut output),
        )
        .unwrap();
    assert_eq!(result, MontyObject::None);
    assert_eq!(output, "Hello world\n");
}

// ===========================================================================
// Setup code with inputs
// ===========================================================================

#[test]
fn session_with_inputs() {
    let runner = MontyRun::new(
        "def scale(x): return x * factor".to_owned(),
        "session_test.py",
        vec!["factor".to_owned()],
    )
    .unwrap();
    let _s = runner
        .into_session_with_print(NoLimitTracker, PrintWriter::Stdout)
        .unwrap();
    // Note: MontyRun::into_session doesn't take inputs directly.
    // The factor variable won't be set since we don't pass inputs to into_session.
    // This test verifies the basic flow works.
    // Calling scale would fail because factor is undefined.
}

// ===========================================================================
// Complex data types
// ===========================================================================

#[test]
fn call_function_returns_list() {
    let mut s = session("def make_list(n): return list(range(n))");
    let result = s.call_function("make_list", vec![MontyObject::Int(3)]).unwrap();
    assert_eq!(
        result,
        MontyObject::List(vec![MontyObject::Int(0), MontyObject::Int(1), MontyObject::Int(2)])
    );
}

#[test]
fn call_function_returns_dict() {
    let mut s = session(
        "\
def make_point(x, y):
    return {'x': x, 'y': y}
",
    );
    let result = s
        .call_function("make_point", vec![MontyObject::Int(1), MontyObject::Int(2)])
        .unwrap();
    if let MontyObject::Dict(pairs) = result {
        assert_eq!(pairs.into_iter().count(), 2);
    } else {
        panic!("expected dict, got: {result:?}");
    }
}

#[test]
fn call_function_many_args() {
    let mut s = session("def sum_all(a, b, c, d, e): return a + b + c + d + e");
    let result = s
        .call_function(
            "sum_all",
            vec![
                MontyObject::Int(1),
                MontyObject::Int(2),
                MontyObject::Int(3),
                MontyObject::Int(4),
                MontyObject::Int(5),
            ],
        )
        .unwrap();
    assert_eq!(result, MontyObject::Int(15));
}

// ===========================================================================
// Setup-time error handling (NameLookup / ExternalCall during module init)
// ===========================================================================

#[test]
fn setup_undefined_name_raises_name_error() {
    // During session setup, referencing an undefined name triggers NameLookup.
    // The session converts this to NameError.
    let runner = MontyRun::new("x = undefined_var".to_owned(), "session_test.py", vec![]).unwrap();
    let err = runner.into_session(NoLimitTracker).unwrap_err();
    assert!(
        err.to_string().contains("name 'undefined_var' is not defined"),
        "got: {err}"
    );
}

#[test]
fn setup_external_call_raises_name_error() {
    // During session setup, calling an external function triggers ExternalCall.
    // The session converts this to NameError since there's no host to resolve it.
    let runner = MontyRun::new("result = some_ext_func(1, 2)".to_owned(), "session_test.py", vec![]).unwrap();
    let err = runner.into_session(NoLimitTracker).unwrap_err();
    assert!(
        err.to_string().contains("name 'some_ext_func' is not defined"),
        "got: {err}"
    );
}

// ===========================================================================
// External function errors from call_function (run_callable coverage)
// ===========================================================================

#[test]
fn call_function_that_calls_undefined_name_fails() {
    // A session function that references an undefined name at call time.
    // This exercises run_callable's NameLookup/ExternalCall error arms in vm/mod.rs.
    let mut s = session("def call_missing(): return unknown_func()");
    let err = s.call_function("call_missing", vec![]).unwrap_err();
    // run_callable catches the NameLookup exit and returns a RuntimeError
    assert!(
        err.to_string().contains("external functions are not supported"),
        "got: {err}"
    );
}

// ===========================================================================
// FunctionDefaults path (call_heap_callable)
// ===========================================================================

#[test]
fn call_function_with_heap_defaults() {
    // When defaults are heap-allocated (e.g. mutable default), the function is stored
    // as FunctionDefaults on the heap. This exercises call_heap_callable's
    // FunctionDefaults branch.
    let mut s = session("def greet(name, greeting='Hi'): return greeting + ' ' + name");
    let result = s
        .call_function("greet", vec![MontyObject::String("Alice".to_owned())])
        .unwrap();
    assert_eq!(result, MontyObject::String("Hi Alice".to_owned()));
}

// ===========================================================================
// convert_args error handling
// ===========================================================================

#[test]
fn convert_args_single_repr_fails() {
    // MontyObject::Repr cannot be converted to a Value, exercising the 1-arg error path.
    let mut s = session("def identity(x): return x");
    let err = s
        .call_function("identity", vec![MontyObject::Repr("bad".to_owned())])
        .unwrap_err();
    assert!(err.to_string().contains("invalid argument type"), "got: {err}");
}

#[test]
fn convert_args_two_second_repr_fails() {
    // Two args where the second is Repr — exercises the 2-arg error cleanup branch
    // where the first arg must be dropped.
    let mut s = session("def add(a, b): return a + b");
    let err = s
        .call_function("add", vec![MontyObject::Int(1), MontyObject::Repr("bad".to_owned())])
        .unwrap_err();
    assert!(err.to_string().contains("invalid argument type"), "got: {err}");
}

#[test]
fn convert_args_two_first_repr_fails() {
    // Two args where the first is Repr — exercises the 2-arg first-arg error path.
    let mut s = session("def add(a, b): return a + b");
    let err = s
        .call_function("add", vec![MontyObject::Repr("bad".to_owned()), MontyObject::Int(1)])
        .unwrap_err();
    assert!(err.to_string().contains("invalid argument type"), "got: {err}");
}

#[test]
fn convert_args_many_middle_repr_fails() {
    // Many args (>2) where one in the middle is Repr — exercises the variadic error
    // cleanup that drains already-converted values.
    let mut s = session("def f(a, b, c, d): return a");
    let err = s
        .call_function(
            "f",
            vec![
                MontyObject::Int(1),
                MontyObject::Int(2),
                MontyObject::Repr("bad".to_owned()),
                MontyObject::Int(4),
            ],
        )
        .unwrap_err();
    assert!(err.to_string().contains("invalid argument type"), "got: {err}");
}

// ===========================================================================
// Builtin callable via session (call_function Builtin branch)
// ===========================================================================

#[test]
fn call_builtin_via_session() {
    // Assigning a builtin function to a global exercises the Builtin branch
    // in call_function when called from run_callable.
    let mut s = session("my_len = len");
    let result = s
        .call_function(
            "my_len",
            vec![MontyObject::List(vec![MontyObject::Int(1), MontyObject::Int(2)])],
        )
        .unwrap();
    assert_eq!(result, MontyObject::Int(2));
}
