//! Persistent execution session for calling Python functions from Rust.
//!
//! [`MontySession`] runs Python code to define functions and variables, then
//! retains the heap, globals, and compiled function table so that individual
//! Python functions can be invoked repeatedly from Rust.
//!
//! # Example
//! ```
//! use monty::{MontyRun, MontyObject, NoLimitTracker};
//!
//! let runner = MontyRun::new(
//!     "def add(a, b): return a + b".to_owned(),
//!     "session.py",
//!     vec![],
//! ).unwrap();
//!
//! let mut session = runner.into_session(NoLimitTracker).unwrap();
//! let result = session.call_function("add", vec![MontyObject::Int(2), MontyObject::Int(3)]).unwrap();
//! assert_eq!(result, MontyObject::Int(5));
//! ```

use std::mem;

use ahash::AHashMap;

use crate::{
    ExcType, MontyException,
    args::ArgValues,
    bytecode::{FrameExit, VM},
    exception_private::RunError,
    heap::{DropWithHeap, Heap},
    io::PrintWriter,
    namespace::NamespaceId,
    object::MontyObject,
    resource::ResourceTracker,
    run::Executor,
    value::Value,
};

/// A persistent execution session that allows calling Python-defined functions from Rust.
///
/// Created by [`MontyRun::into_session`](crate::MontyRun::into_session), which runs
/// the initial Python code (e.g. function definitions) and retains the resulting
/// heap and global namespace. Individual functions can then be called repeatedly
/// via [`call_function`](Self::call_function).
///
/// The session owns all execution state: the heap, globals, interned strings, and
/// compiled bytecode. Functions defined in the initial code remain callable for the
/// lifetime of the session.
///
/// # Type Parameters
/// * `T` — Resource tracker implementation (e.g. `NoLimitTracker` or `LimitedTracker`).
#[derive(Debug)]
pub struct MontySession<T: ResourceTracker> {
    /// The executor containing compiled bytecode and interns.
    executor: Executor,
    /// Maps global variable names to their namespace slot indices.
    name_map: AHashMap<String, NamespaceId>,
    /// Persistent heap across function calls.
    heap: Heap<T>,
    /// Persistent global variable values.
    ///
    /// Functions defined in the initial code are stored here as `Value::DefFunction`,
    /// closures as `Value::Ref` pointing to heap-allocated `Closure` data.
    globals: Vec<Value>,
}

impl<T: ResourceTracker> MontySession<T> {
    /// Creates a new session by running the initial code and retaining state.
    ///
    /// The code is executed to completion (no external function support during setup).
    /// After execution, the heap, globals, and name map are retained so that
    /// functions defined in the code can be called later.
    ///
    /// # Arguments
    /// * `executor` — Compiled code from [`MontyRun`](crate::MontyRun).
    /// * `name_map` — Maps variable names to global namespace slots.
    /// * `resource_tracker` — Resource tracker for limiting memory, time, etc.
    /// * `print` — Writer for print output during setup execution.
    ///
    /// # Errors
    /// Returns `MontyException` if the setup code raises an exception.
    pub(crate) fn new(
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

        // Drop the module return value (we only care about side effects: function defs, variable assignments)
        if let FrameExit::Return(value) = exit {
            value.drop_with_heap(vm.heap);
        }

        // Reclaim globals from the VM
        let globals = vm.take_globals();
        vm.cleanup();

        Ok(Self {
            executor,
            name_map,
            heap,
            globals,
        })
    }

    /// Calls a Python function defined in the session by name.
    ///
    /// Looks up the function in the global namespace, converts the arguments from
    /// `MontyObject` to internal `Value` representations, executes the function,
    /// and converts the result back to `MontyObject`.
    ///
    /// # Arguments
    /// * `name` — The name of the function to call (must exist in global scope).
    /// * `args` — Positional arguments to pass to the function.
    ///
    /// # Errors
    /// Returns `MontyException` if:
    /// - The function name is not found in the global namespace
    /// - The name refers to a non-callable value
    /// - The function raises an exception during execution
    /// - An external function call is encountered (not supported in sync mode)
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
        // Look up the function slot before creating the VM
        let slot_idx = self
            .resolve_callable_slot(name)
            .map_err(|e| e.into_python_exception(&self.executor.interns, &self.executor.code))?;

        // Create a VM with the current state
        let mut vm = VM::new(
            mem::take(&mut self.globals),
            &mut self.heap,
            &self.executor.interns,
            print,
        );

        // Clone the callable from globals with proper refcount handling
        let callable = vm.globals[slot_idx].clone_with_heap(vm.heap);

        // Convert MontyObject args to internal Value args
        let arg_values = match convert_args(args, &mut vm) {
            Ok(av) => av,
            Err(e) => {
                callable.drop_with_heap(vm.heap);
                self.globals = vm.take_globals();
                vm.cleanup();
                return Err(e);
            }
        };

        // Execute the function synchronously
        let result = vm.run_callable(&callable, arg_values);

        // Convert result
        let py_result = match result {
            Ok(value) => Ok(MontyObject::new(value, &mut vm)),
            Err(err) => Err(err),
        };

        // Drop our cloned reference to the callable
        callable.drop_with_heap(vm.heap);

        // Reclaim globals
        self.globals = vm.take_globals();
        vm.cleanup();

        py_result.map_err(|e| e.into_python_exception(&self.executor.interns, &self.executor.code))
    }

    /// Returns a list of all callable function names defined in the session.
    ///
    /// This includes functions, closures, and functions with default arguments.
    /// Does not include builtins or external functions.
    #[must_use]
    pub fn function_names(&self) -> Vec<&str> {
        self.name_map
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
        self.name_map.get(name).is_some_and(|ns_id| {
            let idx = ns_id.index();
            idx < self.globals.len() && is_callable(&self.globals[idx])
        })
    }

    /// Returns a mutable reference to the resource tracker.
    ///
    /// Allows adjusting resource limits between function calls.
    pub fn tracker_mut(&mut self) -> &mut T {
        self.heap.tracker_mut()
    }

    /// Resolves a function name to its global slot index, validating it's callable.
    ///
    /// This does not clone the value — the caller should use `clone_with_heap`
    /// on the globals entry after creating the VM (which provides heap access).
    fn resolve_callable_slot(&self, name: &str) -> Result<usize, RunError> {
        let ns_id = self
            .name_map
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
}

impl<T: ResourceTracker> Drop for MontySession<T> {
    fn drop(&mut self) {
        self.globals.drain(..).drop_with_heap(&mut self.heap);
    }
}

/// Converts `Vec<MontyObject>` to internal `ArgValues`.
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
                        // Clean up already-converted values
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
        Value::DefFunction(_) | Value::Builtin(_) | Value::ExtFunction(_) | Value::ModuleFunction(_) | Value::Ref(_) // Could be a closure or function with defaults
    )
}
