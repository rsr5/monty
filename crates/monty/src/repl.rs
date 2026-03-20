//! Persistent execution session for Monty.
//!
//! [`MontyRepl`] is the unified persistent execution context. It supports two
//! complementary usage patterns:
//!
//! - **REPL mode** — Feed code snippets incrementally via [`feed_run`](MontyRepl::feed_run)
//!   or [`feed_start`](MontyRepl::feed_start). Each snippet is compiled against the
//!   current global state without replaying previous snippets.
//!
//! - **Function-call mode** — Run setup code once (defining functions and variables), then
//!   call individual Python functions repeatedly via [`call_function`](MontyRepl::call_function)
//!   or [`call_function_with_print`](MontyRepl::call_function_with_print).
//!
//! Both modes can be combined on the same session: feed snippets to define functions,
//! then call them later — or call a function that mutates globals, then feed a snippet
//! that reads the updated state.
//!
//! # Example — REPL mode
//! ```
//! use monty::{MontyRepl, MontyObject, NoLimitTracker};
//!
//! let mut session = MontyRepl::new("repl.py", NoLimitTracker);
//! let result = session.feed_run("1 + 2", vec![], monty::PrintWriter::Stdout).unwrap();
//! assert_eq!(result, MontyObject::Int(3));
//! ```
//!
//! # Example — Function-call mode
//! ```
//! use monty::{MontyRun, MontyObject, NoLimitTracker};
//!
//! let runner = MontyRun::new(
//!     "def add(a, b): return a + b".to_owned(),
//!     "session.py",
//!     vec![],
//! ).unwrap();
//!
//! let mut session = runner.into_repl(NoLimitTracker).unwrap();
//! let result = session.call_function("add", vec![MontyObject::Int(2), MontyObject::Int(3)]).unwrap();
//! assert_eq!(result, MontyObject::Int(5));
//! ```

use std::mem;

use ahash::AHashMap;
use ruff_python_ast::token::TokenKind;
use ruff_python_parser::{InterpolatedStringErrorType, LexicalErrorType, ParseErrorType, parse_module};

use crate::{
    ExcType, MontyException,
    args::ArgValues,
    asyncio::CallId,
    bytecode::{Code, Compiler, FrameExit, VM, VMSnapshot},
    exception_private::{RunError, RunResult},
    heap::{DropWithHeap, Heap},
    intern::{InternerBuilder, Interns},
    io::PrintWriter,
    namespace::NamespaceId,
    object::MontyObject,
    os::OsFunction,
    parse::parse_with_interner,
    prepare::prepare_with_existing_names,
    resource::ResourceTracker,
    run::Executor,
    run_progress::{ConvertedExit, ExtFunctionResult, NameLookupResult, convert_frame_exit},
    value::Value,
};

// ---------------------------------------------------------------------------
// MontyRepl
// ---------------------------------------------------------------------------

/// Persistent execution session that preserves heap and global state across
/// multiple operations.
///
/// Supports both REPL-style snippet feeding and direct function calls. Created
/// either via [`MontyRepl::new`] (empty, for REPL use) or via
/// [`MontyRun::into_repl`](crate::MontyRun::into_repl) (pre-seeded with
/// compiled code, for function-call use).
///
/// # Type Parameters
/// * `T` — Resource tracker implementation (e.g. `NoLimitTracker` or `LimitedTracker`).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub struct MontyRepl<T: ResourceTracker> {
    /// Script name used for runtime error messages.
    ///
    /// For REPL sessions, incremental snippets use generated names like
    /// `<python-input-0>` to match CPython's interactive traceback style.
    script_name: String,
    /// Counter for generated `<python-input-N>` snippet filenames (REPL mode).
    next_input_id: u64,
    /// Stable mapping of global variable names to namespace slot IDs.
    global_name_map: AHashMap<String, NamespaceId>,
    /// Persistent intern table across operations so intern/function IDs remain valid.
    interns: Interns,
    /// Persistent heap across operations.
    heap: Heap<T>,
    /// Persistent global variable values.
    ///
    /// Indexed by `NamespaceId` slots from `global_name_map`. Between operations
    /// these are the only VM values that persist — stack and frames are transient.
    globals: Vec<Value>,
}

impl<T: ResourceTracker> MontyRepl<T> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates an empty session with no code parsed or executed.
    ///
    /// All code execution is driven through [`feed_run`](Self::feed_run),
    /// [`feed_start`](Self::feed_start), [`call_function`](Self::call_function),
    /// or [`call_function_with_print`](Self::call_function_with_print).
    #[must_use]
    pub fn new(script_name: &str, resource_tracker: T) -> Self {
        let heap = Heap::new(0, resource_tracker);

        Self {
            script_name: script_name.to_owned(),
            next_input_id: 0,
            global_name_map: AHashMap::new(),
            interns: Interns::new(InternerBuilder::default(), Vec::new()),
            heap,
            globals: Vec::new(),
        }
    }

    /// Creates a session pre-seeded with compiled code and execution state.
    ///
    /// Runs the initial code to completion (defining functions, assigning variables,
    /// etc.), then retains the heap and global namespace so that functions can be
    /// called via [`call_function`](Self::call_function).
    ///
    /// # Arguments
    /// * `executor` — Compiled code from [`MontyRun`](crate::MontyRun).
    /// * `name_map` — Maps variable names to global namespace slots.
    /// * `resource_tracker` — Resource tracker for limiting memory, time, etc.
    /// * `print` — Writer for print output during setup execution.
    ///
    /// # Errors
    /// Returns `MontyException` if the setup code raises an exception.
    pub(crate) fn from_executor(
        executor: Executor,
        name_map: AHashMap<String, NamespaceId>,
        resource_tracker: T,
        print: PrintWriter<'_>,
    ) -> Result<Self, MontyException> {
        let mut heap = Heap::new(executor.namespace_size, resource_tracker);
        let globals = executor.empty_globals();
        let mut vm = VM::new(globals, &mut heap, &executor.interns, print);

        let mut frame_exit_result = vm.run_module(&executor.module_code);

        // Handle NameLookup and ExternalCall exits by raising NameError —
        // during session setup, there's no host to resolve these.
        loop {
            match frame_exit_result {
                Ok(FrameExit::NameLookup { name_id, .. }) => {
                    let name = executor.interns.get_str(name_id);
                    let err = ExcType::name_error(name);
                    frame_exit_result = vm.resume_with_exception(err.into());
                }
                Ok(FrameExit::ExternalCall {
                    function_name,
                    args,
                    name_load_ip,
                    ..
                }) => {
                    if let Some(load_ip) = name_load_ip {
                        vm.set_instruction_ip(load_ip);
                    }
                    let name = function_name.as_str(&executor.interns);
                    args.drop_with_heap(vm.heap);
                    let err = ExcType::name_error(name);
                    frame_exit_result = vm.resume_with_exception(err.into());
                }
                _ => break,
            }
        }

        // Check for runtime errors
        let exit = frame_exit_result.map_err(|e| e.into_python_exception(&executor.interns, &executor.code))?;

        // Drop the module return value (we only care about side effects)
        if let FrameExit::Return(value) = exit {
            value.drop_with_heap(vm.heap);
        }

        let globals = vm.take_globals();
        vm.cleanup();

        Ok(Self {
            script_name: String::new(),
            next_input_id: 0,
            global_name_map: name_map,
            interns: executor.interns,
            heap,
            globals,
        })
    }

    // -----------------------------------------------------------------------
    // REPL mode — feed snippets
    // -----------------------------------------------------------------------

    /// Starts executing a new snippet with pause/resume support.
    ///
    /// This is the session equivalent of `MontyRun::start`: execution may complete,
    /// suspend at external calls / OS calls / unresolved futures, or raise a Python
    /// exception. Resume with the returned state object and eventually recover the
    /// updated session from `ReplProgress::into_complete`.
    ///
    /// Consumes `self` so runtime state can be safely moved into snapshot objects
    /// for serialization and cross-process resume.
    ///
    /// On a Python-level runtime exception the session is **not** destroyed: it is
    /// returned inside `ReplStartError` so the caller can continue feeding
    /// subsequent snippets.
    ///
    /// # Errors
    /// Returns `Err(Box<ReplStartError>)` for syntax, compile-time, or runtime
    /// failures — the session is always preserved inside the error.
    pub fn feed_start(
        self,
        code: &str,
        inputs: Vec<(String, MontyObject)>,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        let mut this = self;
        if code.is_empty() {
            return Ok(ReplProgress::Complete {
                repl: this,
                value: MontyObject::None,
            });
        }

        let (input_names, input_values): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();

        let input_script_name = this.next_input_script_name();
        let executor = match ReplExecutor::new_snippet(
            code.to_owned(),
            &input_script_name,
            this.global_name_map.clone(),
            &this.interns,
            input_names,
        ) {
            Ok(exec) => exec,
            Err(error) => return Err(Box::new(ReplStartError { repl: this, error })),
        };

        this.ensure_globals_size(executor.namespace_size);

        let mut vm = VM::new(mem::take(&mut this.globals), &mut this.heap, &executor.interns, print);

        // Inject inputs with VM alive
        if let Err(error) = inject_inputs_into_vm(&executor, input_values, &mut vm) {
            this.globals = vm.take_globals();
            vm.cleanup();
            return Err(Box::new(ReplStartError { repl: this, error }));
        }

        let vm_result = vm.run_module(&executor.module_code);

        // Convert while VM alive, then snapshot or reclaim globals
        let converted = convert_frame_exit(vm_result, &mut vm);
        if converted.needs_snapshot() {
            let vm_state = vm.snapshot();
            build_repl_progress(converted, Some(vm_state), executor, this)
        } else {
            this.globals = vm.take_globals();
            vm.cleanup();
            build_repl_progress(converted, None, executor, this)
        }
    }

    /// Feeds and executes a new snippet to completion (no pause/resume).
    ///
    /// Compiles only `code` against the current global state, extends the namespace
    /// if new names are introduced, and executes the snippet once. Previously
    /// executed snippets are never replayed.
    ///
    /// # Errors
    /// Returns `MontyException` for syntax/compile/runtime failures.
    pub fn feed_run(
        &mut self,
        code: &str,
        inputs: Vec<(String, MontyObject)>,
        print: PrintWriter<'_>,
    ) -> Result<MontyObject, MontyException> {
        if code.is_empty() {
            return Ok(MontyObject::None);
        }

        let (input_names, input_values): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();

        let input_script_name = self.next_input_script_name();
        let executor = ReplExecutor::new_snippet(
            code.to_owned(),
            &input_script_name,
            self.global_name_map.clone(),
            &self.interns,
            input_names,
        )?;

        self.ensure_globals_size(executor.namespace_size);

        let mut vm = VM::new(mem::take(&mut self.globals), &mut self.heap, &executor.interns, print);

        if let Err(e) = inject_inputs_into_vm(&executor, input_values, &mut vm) {
            self.globals = vm.take_globals();
            vm.cleanup();
            return Err(e);
        }

        let mut frame_exit_result = vm.run_module(&executor.module_code);

        // Handle NameLookup exits by raising NameError — in the non-iterative path,
        // there's no host to resolve names.
        while let Ok(FrameExit::NameLookup { name_id, .. }) = &frame_exit_result {
            let name = executor.interns.get_str(*name_id);
            let err = ExcType::name_error(name);
            frame_exit_result = vm.resume_with_exception(err.into());
        }

        // Convert output while VM alive
        let result = frame_exit_to_object(frame_exit_result, &mut vm);

        self.globals = vm.take_globals();
        vm.cleanup();

        // Commit compiler metadata even on runtime errors.
        let ReplExecutor {
            name_map,
            interns,
            code,
            ..
        } = executor;
        self.global_name_map = name_map;
        self.interns = interns;

        result.map_err(|e| e.into_python_exception(&self.interns, &code))
    }

    // -----------------------------------------------------------------------
    // Function-call mode
    // -----------------------------------------------------------------------

    /// Calls a Python function defined in the session by name.
    ///
    /// Looks up the function in the global namespace, converts the arguments,
    /// executes the function, and converts the result back.
    ///
    /// # Errors
    /// Returns `MontyException` if the function is not found, not callable,
    /// raises an exception, or encounters an external function call.
    pub fn call_function(&mut self, name: &str, args: Vec<MontyObject>) -> Result<MontyObject, MontyException> {
        self.call_function_with_print(name, args, PrintWriter::Stdout)
    }

    /// Calls a Python function with a custom print writer.
    ///
    /// Same as [`call_function`](Self::call_function) but allows capturing print output.
    pub fn call_function_with_print(
        &mut self,
        name: &str,
        args: Vec<MontyObject>,
        print: PrintWriter<'_>,
    ) -> Result<MontyObject, MontyException> {
        let slot_idx = self
            .resolve_callable_slot(name)
            .map_err(|e| e.into_python_exception(&self.interns, ""))?;

        let mut vm = VM::new(mem::take(&mut self.globals), &mut self.heap, &self.interns, print);

        let callable = vm.globals[slot_idx].clone_with_heap(vm.heap);

        let arg_values = match convert_args(args, &mut vm) {
            Ok(av) => av,
            Err(e) => {
                callable.drop_with_heap(vm.heap);
                self.globals = vm.take_globals();
                vm.cleanup();
                return Err(e);
            }
        };

        let result = vm.run_callable(&callable, arg_values);

        let py_result = match result {
            Ok(value) => Ok(MontyObject::new(value, &mut vm)),
            Err(err) => Err(err),
        };

        callable.drop_with_heap(vm.heap);
        self.globals = vm.take_globals();
        vm.cleanup();

        py_result.map_err(|e| e.into_python_exception(&self.interns, ""))
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Returns a list of all callable function names defined in the session.
    ///
    /// Includes functions, closures, and functions with default arguments.
    /// Does not include builtins or external functions.
    #[must_use]
    pub fn function_names(&self) -> Vec<&str> {
        self.global_name_map
            .iter()
            .filter_map(|(name, ns_id)| {
                let idx = ns_id.index();
                if idx < self.globals.len() && is_callable(&self.globals[idx]) {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns whether a function with the given name exists in the session.
    #[must_use]
    pub fn has_function(&self, name: &str) -> bool {
        self.global_name_map.get(name).is_some_and(|ns_id| {
            let idx = ns_id.index();
            idx < self.globals.len() && is_callable(&self.globals[idx])
        })
    }

    /// Returns a mutable reference to the resource tracker.
    ///
    /// Allows adjusting resource limits between operations.
    pub fn tracker_mut(&mut self) -> &mut T {
        self.heap.tracker_mut()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Resolves a function name to its global slot index, validating it's callable.
    fn resolve_callable_slot(&self, name: &str) -> Result<usize, RunError> {
        let ns_id = self
            .global_name_map
            .get(name)
            .ok_or_else(|| -> RunError { ExcType::name_error(name).into() })?;

        let idx = ns_id.index();
        if idx >= self.globals.len() {
            return Err(ExcType::name_error(name).into());
        }

        let value = &self.globals[idx];
        if matches!(value, Value::Undefined) {
            return Err(ExcType::name_error(name).into());
        }

        if !is_callable(value) {
            return Err(ExcType::type_error(format!("'{name}' is not callable")));
        }

        Ok(idx)
    }

    /// Grows the globals vector to at least `size` slots.
    fn ensure_globals_size(&mut self, size: usize) {
        if self.globals.len() < size {
            self.globals.resize_with(size, || Value::Undefined);
        }
    }

    /// Returns the generated filename for the next interactive snippet.
    fn next_input_script_name(&mut self) -> String {
        let input_id = self.next_input_id;
        self.next_input_id += 1;
        format!("<python-input-{input_id}>")
    }
}

impl<T: ResourceTracker + serde::Serialize> MontyRepl<T> {
    /// Serializes the session state to bytes.
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn dump(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }
}

impl<T: ResourceTracker + serde::de::DeserializeOwned> MontyRepl<T> {
    /// Restores a session from bytes produced by [`dump`](Self::dump).
    ///
    /// # Errors
    /// Returns an error if deserialization fails.
    pub fn load(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}

impl<T: ResourceTracker> Drop for MontyRepl<T> {
    fn drop(&mut self) {
        self.globals.drain(..).drop_with_heap(&mut self.heap);
    }
}

// ---------------------------------------------------------------------------
// ReplProgress and per-variant structs
// ---------------------------------------------------------------------------

/// Result of a single suspendable session operation (snippet or function call).
///
/// Each variant (except `Complete`) wraps a dedicated struct that owns the
/// execution state and exposes resume methods. On completion, the updated
/// session is returned alongside the result so callers can continue using it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub enum ReplProgress<T: ResourceTracker> {
    /// Execution paused at an external function call or dataclass method call.
    FunctionCall(ReplFunctionCall<T>),
    /// Execution paused for an OS-level operation.
    OsCall(ReplOsCall<T>),
    /// All async tasks are blocked waiting for external futures to resolve.
    ResolveFutures(ReplResolveFutures<T>),
    /// Execution paused for an unresolved name lookup.
    NameLookup(ReplNameLookup<T>),
    /// Execution completed with the updated session and result value.
    Complete {
        /// Updated session state — ready for further operations.
        repl: MontyRepl<T>,
        /// Final result produced by the operation.
        value: MontyObject,
    },
}

/// Error returned when an operation raises a Python exception during
/// `feed_start()` or `resume()`.
///
/// The session is always preserved inside the error so the caller can inspect
/// the exception and continue using the session.
#[derive(Debug)]
pub struct ReplStartError<T: ResourceTracker> {
    /// REPL state after the failed operation — ready for further use.
    pub repl: MontyRepl<T>,
    /// The Python exception that was raised.
    pub error: MontyException,
}

impl<T: ResourceTracker> ReplProgress<T> {
    /// Consumes the progress and returns the `ReplFunctionCall` struct.
    #[must_use]
    pub fn into_function_call(self) -> Option<ReplFunctionCall<T>> {
        match self {
            Self::FunctionCall(call) => Some(call),
            _ => None,
        }
    }

    /// Consumes the progress and returns the `ReplResolveFutures` struct.
    #[must_use]
    pub fn into_resolve_futures(self) -> Option<ReplResolveFutures<T>> {
        match self {
            Self::ResolveFutures(state) => Some(state),
            _ => None,
        }
    }

    /// Consumes the progress and returns the `ReplNameLookup` struct.
    #[must_use]
    pub fn into_name_lookup(self) -> Option<ReplNameLookup<T>> {
        match self {
            Self::NameLookup(lookup) => Some(lookup),
            _ => None,
        }
    }

    /// Consumes the progress and returns the completed session and value.
    #[must_use]
    pub fn into_complete(self) -> Option<(MontyRepl<T>, MontyObject)> {
        match self {
            Self::Complete { repl, value } => Some((repl, value)),
            _ => None,
        }
    }

    /// Extracts the session from any progress variant, discarding the in-flight
    /// execution state.
    ///
    /// Use this to recover the session when you need to abandon the current
    /// operation. The session state reflects any mutations that occurred before
    /// the snapshot was taken.
    #[must_use]
    pub fn into_repl(self) -> MontyRepl<T> {
        match self {
            Self::FunctionCall(call) => call.into_repl(),
            Self::OsCall(call) => call.into_repl(),
            Self::ResolveFutures(state) => state.into_repl(),
            Self::NameLookup(lookup) => lookup.into_repl(),
            Self::Complete { repl, .. } => repl,
        }
    }
}

impl<T: ResourceTracker + serde::Serialize> ReplProgress<T> {
    /// Serializes the session execution progress to binary format.
    ///
    /// # Errors
    /// Returns an error if serialization fails.
    pub fn dump(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }
}

impl<T: ResourceTracker + serde::de::DeserializeOwned> ReplProgress<T> {
    /// Deserializes session execution progress from binary format.
    ///
    /// # Errors
    /// Returns an error if deserialization fails.
    pub fn load(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}

// ---------------------------------------------------------------------------
// ReplFunctionCall
// ---------------------------------------------------------------------------

/// Execution paused at an external function call or dataclass method call.
///
/// Resume with `resume(result, print)` to provide the return value and continue,
/// or `resume_pending(print)` to push an `ExternalFuture` for async resolution.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub struct ReplFunctionCall<T: ResourceTracker> {
    /// The name of the function or method being called.
    pub function_name: String,
    /// The positional arguments passed to the function.
    pub args: Vec<MontyObject>,
    /// The keyword arguments passed to the function (key, value pairs).
    pub kwargs: Vec<(MontyObject, MontyObject)>,
    /// Unique identifier for this call (used for async correlation).
    pub call_id: u32,
    /// Whether this is a dataclass method call (first arg is `self`).
    pub method_call: bool,
    /// Internal session execution snapshot.
    snapshot: ReplSnapshot<T>,
}

impl<T: ResourceTracker> ReplFunctionCall<T> {
    /// Extracts the session, discarding the in-flight execution state.
    ///
    /// Restores globals from the VM snapshot so the session remains usable.
    #[must_use]
    pub fn into_repl(self) -> MontyRepl<T> {
        self.snapshot.into_repl()
    }

    /// Resumes execution with an external result.
    pub fn resume(
        self,
        result: impl Into<ExtFunctionResult>,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        self.snapshot.run(result, print)
    }

    /// Resumes execution by pushing an `ExternalFuture` for async resolution.
    ///
    /// Uses `self.call_id` internally — no need to pass it again.
    pub fn resume_pending(self, print: PrintWriter<'_>) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        self.snapshot.run(ExtFunctionResult::Future(self.call_id), print)
    }
}

// ---------------------------------------------------------------------------
// ReplOsCall
// ---------------------------------------------------------------------------

/// Execution paused for an OS-level operation.
///
/// Resume with `resume(result, print)` to provide the OS call result and continue.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub struct ReplOsCall<T: ResourceTracker> {
    /// The OS function to execute.
    pub function: OsFunction,
    /// The positional arguments for the OS function.
    pub args: Vec<MontyObject>,
    /// The keyword arguments passed to the function (key, value pairs).
    pub kwargs: Vec<(MontyObject, MontyObject)>,
    /// Unique identifier for this call (used for async correlation).
    pub call_id: u32,
    /// Internal session execution snapshot.
    snapshot: ReplSnapshot<T>,
}

impl<T: ResourceTracker> ReplOsCall<T> {
    /// Extracts the session, discarding the in-flight execution state.
    #[must_use]
    pub fn into_repl(self) -> MontyRepl<T> {
        self.snapshot.into_repl()
    }

    /// Resumes execution with the OS call result.
    pub fn resume(
        self,
        result: impl Into<ExtFunctionResult>,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        self.snapshot.run(result, print)
    }
}

// ---------------------------------------------------------------------------
// ReplNameLookup
// ---------------------------------------------------------------------------

/// Execution paused for an unresolved name lookup.
///
/// The host should check if the name corresponds to a known external function or
/// value. Call `resume(result, print)` with the appropriate `NameLookupResult`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub struct ReplNameLookup<T: ResourceTracker> {
    /// The name being looked up.
    pub name: String,
    /// The namespace slot where the resolved value should be cached.
    namespace_slot: u16,
    /// Whether this is a global slot or a local/function slot.
    is_global: bool,
    /// Internal session execution snapshot.
    snapshot: ReplSnapshot<T>,
}

impl<T: ResourceTracker> ReplNameLookup<T> {
    /// Extracts the session, discarding the in-flight execution state.
    #[must_use]
    pub fn into_repl(self) -> MontyRepl<T> {
        self.snapshot.into_repl()
    }

    /// Resumes execution after name resolution.
    ///
    /// Caches the resolved value in the namespace slot before restoring the VM,
    /// then either pushes the value onto the stack or raises `NameError`.
    pub fn resume(
        self,
        result: NameLookupResult,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        let Self {
            name,
            namespace_slot,
            is_global,
            snapshot,
        } = self;

        let ReplSnapshot {
            mut repl,
            executor,
            vm_state,
        } = snapshot;

        let mut vm = VM::restore(
            vm_state,
            &executor.module_code,
            &mut repl.heap,
            &executor.interns,
            print,
        );

        let vm_result = match result {
            NameLookupResult::Value(obj) => {
                let value = match obj.to_value(&mut vm) {
                    Ok(v) => v,
                    Err(e) => {
                        repl.globals = vm.take_globals();
                        vm.cleanup();
                        let error = MontyException::runtime_error(format!("invalid name lookup result: {e}"));
                        return Err(Box::new(ReplStartError { repl, error }));
                    }
                };

                // Cache the resolved value in the appropriate slot
                let slot = namespace_slot as usize;
                if is_global {
                    let cloned = value.clone_with_heap(vm.heap);
                    let old = mem::replace(&mut vm.globals[slot], cloned);
                    old.drop_with_heap(vm.heap);
                } else {
                    let stack_base = vm.current_stack_base();
                    let cloned = value.clone_with_heap(vm.heap);
                    let old = mem::replace(&mut vm.stack[stack_base + slot], cloned);
                    old.drop_with_heap(vm.heap);
                }

                vm.push(value);
                vm.run()
            }
            NameLookupResult::Undefined => {
                let err: RunError = ExcType::name_error(&name).into();
                vm.resume_with_exception(err)
            }
        };

        let converted = convert_frame_exit(vm_result, &mut vm);
        if converted.needs_snapshot() {
            let vm_state = vm.snapshot();
            build_repl_progress(converted, Some(vm_state), executor, repl)
        } else {
            repl.globals = vm.take_globals();
            vm.cleanup();
            build_repl_progress(converted, None, executor, repl)
        }
    }
}

// ---------------------------------------------------------------------------
// ReplResolveFutures
// ---------------------------------------------------------------------------

/// Execution state blocked on unresolved external futures.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub struct ReplResolveFutures<T: ResourceTracker> {
    /// Persistent session state while suspended.
    repl: MontyRepl<T>,
    /// Compiled snippet and intern/function tables for this execution.
    executor: ReplExecutor,
    /// VM stack/frame state at suspension.
    vm_state: VMSnapshot,
    /// Pending call IDs expected by this snapshot.
    pending_call_ids: Vec<u32>,
}

impl<T: ResourceTracker> ReplResolveFutures<T> {
    /// Extracts the session, discarding the in-flight execution state.
    #[must_use]
    pub fn into_repl(self) -> MontyRepl<T> {
        self.repl
    }

    /// Returns unresolved call IDs for this suspended state.
    #[must_use]
    pub fn pending_call_ids(&self) -> &[u32] {
        &self.pending_call_ids
    }

    /// Resumes execution with zero or more resolved futures.
    ///
    /// Supports incremental resolution: callers can provide a subset of pending
    /// call IDs and continue resolving over multiple resumes.
    pub fn resume(
        self,
        results: Vec<(u32, ExtFunctionResult)>,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        let Self {
            mut repl,
            executor,
            vm_state,
            pending_call_ids,
        } = self;

        let invalid_call_id = results
            .iter()
            .find(|(call_id, _)| !pending_call_ids.contains(call_id))
            .map(|(call_id, _)| *call_id);

        let mut vm = VM::restore(
            vm_state,
            &executor.module_code,
            &mut repl.heap,
            &executor.interns,
            print,
        );

        if let Some(call_id) = invalid_call_id {
            repl.globals = vm.take_globals();
            vm.cleanup();
            let error = MontyException::runtime_error(format!(
                "unknown call_id {call_id}, expected one of: {pending_call_ids:?}"
            ));
            return Err(Box::new(ReplStartError { repl, error }));
        }

        for (call_id, ext_result) in results {
            match ext_result {
                ExtFunctionResult::Return(obj) => {
                    if let Err(e) = vm.resolve_future(call_id, obj) {
                        repl.globals = vm.take_globals();
                        vm.cleanup();
                        let error =
                            MontyException::runtime_error(format!("Invalid return type for call {call_id}: {e}"));
                        return Err(Box::new(ReplStartError { repl, error }));
                    }
                }
                ExtFunctionResult::Error(exc) => vm.fail_future(call_id, RunError::from(exc)),
                ExtFunctionResult::Future(_) => {}
                ExtFunctionResult::NotFound(function_name) => {
                    vm.fail_future(call_id, ExtFunctionResult::not_found_exc(&function_name));
                }
            }
        }

        if let Some(error) = vm.take_failed_task_error() {
            repl.globals = vm.take_globals();
            vm.cleanup();
            let error = error.into_python_exception(&executor.interns, &executor.code);
            return Err(Box::new(ReplStartError { repl, error }));
        }

        let main_task_ready = vm.prepare_current_task_after_resolve();

        let loaded_task = match vm.load_ready_task_if_needed() {
            Ok(loaded) => loaded,
            Err(e) => {
                repl.globals = vm.take_globals();
                vm.cleanup();
                let error = e.into_python_exception(&executor.interns, &executor.code);
                return Err(Box::new(ReplStartError { repl, error }));
            }
        };

        if !main_task_ready && !loaded_task {
            let pending_call_ids = vm.get_pending_call_ids();
            if !pending_call_ids.is_empty() {
                let vm_state = vm.snapshot();
                let pending_call_ids: Vec<u32> = pending_call_ids.iter().map(|id| id.raw()).collect();
                return Ok(ReplProgress::ResolveFutures(Self {
                    repl,
                    executor,
                    vm_state,
                    pending_call_ids,
                }));
            }
        }

        let vm_result = vm.run();

        let converted = convert_frame_exit(vm_result, &mut vm);
        if converted.needs_snapshot() {
            let vm_state = vm.snapshot();
            build_repl_progress(converted, Some(vm_state), executor, repl)
        } else {
            repl.globals = vm.take_globals();
            vm.cleanup();
            build_repl_progress(converted, None, executor, repl)
        }
    }
}

// ---------------------------------------------------------------------------
// REPL continuation mode (public utility)
// ---------------------------------------------------------------------------

/// Parse-derived continuation state for interactive REPL input collection.
///
/// Used by interactive REPL frontends to decide whether to execute the buffered
/// snippet immediately, keep collecting continuation lines, or require a
/// terminating blank line for block statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplContinuationMode {
    /// The current snippet is syntactically complete and can run now.
    Complete,
    /// The snippet is incomplete and needs more continuation lines.
    IncompleteImplicit,
    /// The snippet opened an indented block and should wait for a trailing blank
    /// line before execution.
    IncompleteBlock,
}

/// Detects whether REPL source is complete or needs more input.
///
/// Mirrors CPython's broad interactive behavior:
/// - Incomplete bracketed / parenthesized / triple-quoted constructs continue.
/// - Clause headers (`if:`, `def:`, etc.) require an indented body and then a
///   terminating blank line before execution.
/// - All other parse outcomes are treated as complete.
#[must_use]
pub fn detect_repl_continuation_mode(source: &str) -> ReplContinuationMode {
    let Err(error) = parse_module(source) else {
        return ReplContinuationMode::Complete;
    };

    match error.error {
        ParseErrorType::OtherError(msg) => {
            if msg.starts_with("Expected an indented block after ") {
                ReplContinuationMode::IncompleteBlock
            } else {
                ReplContinuationMode::Complete
            }
        }
        ParseErrorType::Lexical(LexicalErrorType::Eof)
        | ParseErrorType::ExpectedToken {
            found: TokenKind::EndOfFile,
            ..
        }
        | ParseErrorType::FStringError(InterpolatedStringErrorType::UnterminatedTripleQuotedString)
        | ParseErrorType::TStringError(InterpolatedStringErrorType::UnterminatedTripleQuotedString) => {
            ReplContinuationMode::IncompleteImplicit
        }
        _ => ReplContinuationMode::Complete,
    }
}

// ---------------------------------------------------------------------------
// ReplExecutor — internal compilation helper
// ---------------------------------------------------------------------------

/// Compiled snippet representation used by session execution.
///
/// Mirrors the data shape needed by VM execution but supports incremental
/// compilation with seeded interns and name maps.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ReplExecutor {
    /// Number of slots needed in the global namespace.
    namespace_size: usize,
    /// Maps variable names to their indices in the namespace.
    name_map: AHashMap<String, NamespaceId>,
    /// Compiled bytecode for the snippet.
    module_code: Code,
    /// Interned strings and compiled functions for this snippet.
    interns: Interns,
    /// Source code used for traceback/error rendering.
    code: String,
    /// Input variable names that were injected for this snippet.
    input_names: Vec<String>,
}

impl ReplExecutor {
    /// Compiles one snippet against existing session metadata.
    ///
    /// Seeds parsing from existing interns so old `StringId` values stay stable,
    /// reuses existing name map and appends new global names only.
    fn new_snippet(
        code: String,
        script_name: &str,
        mut existing_name_map: AHashMap<String, NamespaceId>,
        existing_interns: &Interns,
        input_names: Vec<String>,
    ) -> Result<Self, MontyException> {
        // Pre-register input names so they get stable slots before preparation.
        for name in &input_names {
            let next_slot = existing_name_map.len();
            existing_name_map
                .entry(name.clone())
                .or_insert_with(|| NamespaceId::new(next_slot));
        }

        let seeded_interner = InternerBuilder::from_interns(existing_interns, &code);
        let parse_result = parse_with_interner(&code, script_name, seeded_interner)
            .map_err(|e| e.into_python_exc(script_name, &code))?;
        let prepared = prepare_with_existing_names(parse_result, existing_name_map)
            .map_err(|e| e.into_python_exc(script_name, &code))?;

        let existing_functions = existing_interns.functions_clone();
        let mut interns = Interns::new(prepared.interner, Vec::new());
        let namespace_size_u16 = u16::try_from(prepared.namespace_size).expect("module namespace size exceeds u16");
        let compile_result =
            Compiler::compile_module_with_functions(&prepared.nodes, &interns, namespace_size_u16, existing_functions)
                .map_err(|e| e.into_python_exc(script_name, &code))?;
        interns.set_functions(compile_result.functions);

        Ok(Self {
            namespace_size: prepared.namespace_size,
            name_map: prepared.name_map,
            module_code: compile_result.code,
            interns,
            code,
            input_names,
        })
    }
}

// ---------------------------------------------------------------------------
// ReplSnapshot — internal execution state for suspend/resume
// ---------------------------------------------------------------------------

/// Execution state that can be resumed after an external call.
///
/// This is `pub(crate)` — callers interact with the per-variant structs
/// (`ReplFunctionCall`, etc.).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::de::DeserializeOwned"))]
pub(crate) struct ReplSnapshot<T: ResourceTracker> {
    /// Persistent session state while suspended.
    repl: MontyRepl<T>,
    /// Compiled snippet and intern/function tables for this execution.
    executor: ReplExecutor,
    /// VM stack/frame state at suspension.
    vm_state: VMSnapshot,
}

impl<T: ResourceTracker> ReplSnapshot<T> {
    /// Extracts the session, restoring globals from the VM snapshot.
    fn into_repl(self) -> MontyRepl<T> {
        let Self { mut repl, vm_state, .. } = self;
        repl.globals = vm_state.globals;
        repl
    }

    /// Continues execution with an external result.
    fn run(
        self,
        result: impl Into<ExtFunctionResult>,
        print: PrintWriter<'_>,
    ) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
        let Self {
            mut repl,
            executor,
            vm_state,
        } = self;

        let ext_result = result.into();

        let mut vm = VM::restore(
            vm_state,
            &executor.module_code,
            &mut repl.heap,
            &executor.interns,
            print,
        );

        let vm_result = match ext_result {
            ExtFunctionResult::Return(obj) => vm.resume(obj),
            ExtFunctionResult::Error(exc) => vm.resume_with_exception(exc.into()),
            ExtFunctionResult::Future(raw_call_id) => {
                let call_id = CallId::new(raw_call_id);
                vm.add_pending_call(call_id);
                vm.push(Value::ExternalFuture(call_id));
                vm.run()
            }
            ExtFunctionResult::NotFound(function_name) => {
                vm.resume_with_exception(ExtFunctionResult::not_found_exc(&function_name))
            }
        };

        let converted = convert_frame_exit(vm_result, &mut vm);
        if converted.needs_snapshot() {
            let vm_state = vm.snapshot();
            build_repl_progress(converted, Some(vm_state), executor, repl)
        } else {
            repl.globals = vm.take_globals();
            vm.cleanup();
            build_repl_progress(converted, None, executor, repl)
        }
    }
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

/// Injects input values into the VM's global namespace slots.
fn inject_inputs_into_vm(
    executor: &ReplExecutor,
    input_values: Vec<MontyObject>,
    vm: &mut VM<'_, '_, impl ResourceTracker>,
) -> Result<(), MontyException> {
    for (name, obj) in executor.input_names.iter().zip(input_values) {
        let slot = executor
            .name_map
            .get(name)
            .expect("input name should have a namespace slot")
            .index();
        let value = obj
            .to_value(vm)
            .map_err(|e| MontyException::runtime_error(format!("invalid input type: {e}")))?;
        let old = mem::replace(&mut vm.globals[slot], value);
        old.drop_with_heap(vm.heap);
    }
    Ok(())
}

/// Converts module/frame exit results into plain `MontyObject` outputs.
///
/// Used by the non-iterative `feed_run` path where suspendable outcomes are not
/// supported and should produce errors.
fn frame_exit_to_object(
    frame_exit_result: RunResult<FrameExit>,
    vm: &mut VM<'_, '_, impl ResourceTracker>,
) -> RunResult<MontyObject> {
    match frame_exit_result? {
        FrameExit::Return(return_value) => Ok(MontyObject::new(return_value, vm)),
        FrameExit::ExternalCall {
            function_name, args, ..
        } => {
            args.drop_with_heap(vm.heap);
            let function_name = function_name.as_str(vm.interns);
            Err(ExcType::not_implemented(format!(
                "External function '{function_name}' not implemented with standard execution"
            ))
            .into())
        }
        FrameExit::OsCall { function, args, .. } => {
            args.drop_with_heap(vm.heap);
            Err(ExcType::not_implemented(format!(
                "OS function '{function}' not implemented with standard execution"
            ))
            .into())
        }
        FrameExit::MethodCall { method_name, args, .. } => {
            args.drop_with_heap(vm.heap);
            let name = method_name.as_str(vm.interns);
            Err(
                ExcType::not_implemented(format!("Method call '{name}' not implemented with standard execution"))
                    .into(),
            )
        }
        FrameExit::ResolveFutures(_) => {
            Err(ExcType::not_implemented("async futures not supported by standard execution.").into())
        }
        FrameExit::NameLookup { name_id, .. } => {
            let name = vm.interns.get_str(name_id);
            Err(ExcType::name_error(name).into())
        }
    }
}

/// Assembles a `ReplProgress` from already-converted data.
///
/// On completion/error, compiler metadata is committed to the session so
/// subsequent operations see updated intern tables and name maps.
fn build_repl_progress<T: ResourceTracker>(
    converted: ConvertedExit,
    vm_state: Option<VMSnapshot>,
    executor: ReplExecutor,
    mut repl: MontyRepl<T>,
) -> Result<ReplProgress<T>, Box<ReplStartError<T>>> {
    macro_rules! new_repl_snapshot {
        () => {
            ReplSnapshot {
                repl,
                executor,
                vm_state: vm_state.expect("snapshot should exist"),
            }
        };
    }

    match converted {
        ConvertedExit::Complete(obj) => {
            let ReplExecutor { name_map, interns, .. } = executor;
            repl.global_name_map = name_map;
            repl.interns = interns;
            Ok(ReplProgress::Complete { repl, value: obj })
        }
        ConvertedExit::FunctionCall {
            function_name,
            args,
            kwargs,
            call_id,
            method_call,
        } => Ok(ReplProgress::FunctionCall(ReplFunctionCall {
            function_name,
            args,
            kwargs,
            call_id,
            method_call,
            snapshot: new_repl_snapshot!(),
        })),
        ConvertedExit::OsCall {
            function,
            args,
            kwargs,
            call_id,
        } => Ok(ReplProgress::OsCall(ReplOsCall {
            function,
            args,
            kwargs,
            call_id,
            snapshot: new_repl_snapshot!(),
        })),
        ConvertedExit::ResolveFutures(pending_call_ids) => Ok(ReplProgress::ResolveFutures(ReplResolveFutures {
            repl,
            executor,
            vm_state: vm_state.expect("snapshot should exist for ResolveFutures"),
            pending_call_ids,
        })),
        ConvertedExit::NameLookup {
            name,
            namespace_slot,
            is_global,
        } => Ok(ReplProgress::NameLookup(ReplNameLookup {
            name,
            namespace_slot,
            is_global,
            snapshot: new_repl_snapshot!(),
        })),
        ConvertedExit::Error(err) => {
            let error = err.into_python_exception(&executor.interns, &executor.code);
            let ReplExecutor { name_map, interns, .. } = executor;
            repl.global_name_map = name_map;
            repl.interns = interns;
            Err(Box::new(ReplStartError { repl, error }))
        }
    }
}

/// Converts `Vec<MontyObject>` to internal `ArgValues` for function calls.
fn convert_args(
    args: Vec<MontyObject>,
    vm: &mut VM<'_, '_, impl ResourceTracker>,
) -> Result<ArgValues, MontyException> {
    match args.len() {
        0 => Ok(ArgValues::Empty),
        1 => {
            let value = args
                .into_iter()
                .next()
                .expect("checked len")
                .to_value(vm)
                .map_err(|e| MontyException::runtime_error(format!("invalid argument type: {e}")))?;
            Ok(ArgValues::One(value))
        }
        2 => {
            let mut iter = args.into_iter();
            let a = iter
                .next()
                .expect("checked len")
                .to_value(vm)
                .map_err(|e| MontyException::runtime_error(format!("invalid argument type: {e}")))?;
            match iter.next().expect("checked len").to_value(vm) {
                Ok(b) => Ok(ArgValues::Two(a, b)),
                Err(e) => {
                    a.drop_with_heap(vm.heap);
                    Err(MontyException::runtime_error(format!("invalid argument type: {e}")))
                }
            }
        }
        _ => {
            let mut values = Vec::with_capacity(args.len());
            for arg in args {
                match arg.to_value(vm) {
                    Ok(value) => values.push(value),
                    Err(e) => {
                        values.drain(..).drop_with_heap(vm.heap);
                        return Err(MontyException::runtime_error(format!("invalid argument type: {e}")));
                    }
                }
            }
            Ok(ArgValues::ArgsKargs {
                args: values,
                kwargs: crate::args::KwargsValues::Empty,
            })
        }
    }
}

/// Returns `true` if the value is a callable type.
fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::DefFunction(_) | Value::Builtin(_) | Value::ExtFunction(_) | Value::ModuleFunction(_) | Value::Ref(_)
    )
}

// ---------------------------------------------------------------------------
// Type alias for function-call mode
// ---------------------------------------------------------------------------

/// Type alias for `MontyRepl` — the unified session type supports both REPL
/// snippet feeding and direct function calls.
pub type MontySession<T> = MontyRepl<T>;
