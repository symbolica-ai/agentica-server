use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::component::ResourceTable;
use wasmtime::{Config, Engine, Error, Store, component::*};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use wasmtime_wasi::p2::add_to_linker_async;
use wasmtime_wasi_io::IoView;

wasmtime::component::bindgen!({ path: "../wit/", world: "env", imports: { default: async }, exports: { default: async } });

struct Imports {
    recv_bytes: PyObject,
    send_bytes: PyObject,
    recv_ready: PyObject,
    write_log: PyObject
}

struct Ctx {
    table: ResourceTable,
    wasi: WasiCtx,
    /* wit imports */
    imports: Imports,
}

impl IoView for Ctx {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

impl WasiView for Ctx {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView { ctx: &mut self.wasi, table: &mut self.table }
    }
}

fn pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn pyerr_to_wasmtime_err(e: PyErr) -> wasmtime::Error {
    let msg = Python::with_gil(|py| {
        let ty = e.get_type(py);
        let val = e.value(py);

        // Prefer Python-side formatting: "TypeError: message\n"
        if let Ok(tbmod) = py.import("traceback") {
            if let Ok(list_obj) = tbmod.call_method1("format_exception_only", (&ty, &val)) {
                if let Ok(parts) = list_obj.extract::<Vec<String>>() {
                    return parts.concat();
                }
            }
        }

        // Fallback: "Type: message", both owned strings
        let ty_name = ty
            .name()
            .ok()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "PyErr".to_string());
        let val_str = val
            .str()
            .ok()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<error>".to_string());
        format!("{ty_name}: {val_str}")
    });
    wasmtime::Error::msg(msg)
}

struct WasmData {
    store: Store<Ctx>,
    comp: Component,
    linker: Linker<Ctx>,
    env: Option<Env>,
    logging: bool,
    log_tags: Option<String>,
    id_name: String,
}

impl WasmData {
    async fn instantiate(&mut self) {
        if self.env.is_none() {
            if self.logging {
                eprintln!("WASMRunner: instantiating");
            }
            self.env = match Env::instantiate_async(&mut self.store, &self.comp, &self.linker).await
            {
                Ok(env) => {
                    if self.logging {
                        eprintln!("WASMRunner: calling init_exec_env");
                    }
                    let init_res = env
                        .call_init_exec_env(
                            &mut self.store,
                            &self.id_name,
                            self.log_tags.as_deref(),
                        )
                        .await;
                    match init_res {
                        Ok(()) => Some(env),
                        Err(e) => {
                            eprintln!("WASMRunner: init_exec_env failed: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("WASMRunner: failed to instantiate: {}", e);
                    None
                }
            };
        }
    }

    async fn run_msg_loop(&mut self) -> Result<(), Error> {
        if self.logging {
            eprintln!("WASMRunner: run_msg_loop()");
        }
        let res = match &self.env {
            Some(env) => env.call_run_msg_loop(&mut self.store).await.into(),
            None => Err(Error::msg("WASMRunner: not started")),
        };
        if self.logging {
            if res.is_err() {
                eprintln!("WASMRunner: run_msg_loop() returned error");
            } else {
                eprintln!("WASMRunner: run_msg_loop() finished normally");
            }
        };
        return res;
    }
}

#[pyclass]
struct WasmRunner {
    wasm: Arc<Mutex<WasmData>>,
    logging: bool,
}

impl WasmRunner {
    fn is_running(&self) -> bool {
        match self.wasm.try_lock() {
            Ok(_) => false,
            Err(_) => true,
        }
    }
}

#[pymethods]
impl WasmRunner {
    #[new]
    #[pyo3(signature = (
        id_name,
        send_bytes,
        recv_bytes,
        recv_ready,
        write_log,
        log_tags=None,
        wasm_inherit_io=true,
        wasm_path=None,
        wasm_compiled_cache=None,
        runner_logging=false,
    ))]
    fn new(
        _py: Python<'_>,
        id_name: String,
        send_bytes: PyObject,
        recv_bytes: PyObject,
        recv_ready: PyObject,
        write_log: PyObject,
        log_tags: Option<String>,
        wasm_inherit_io: bool,
        wasm_path: Option<String>,
        wasm_compiled_cache: Option<String>,
        runner_logging: bool,
    ) -> PyResult<Self> {
        if runner_logging {
            eprintln!("WASMRunner: new()");
        }
        let imports = Imports {
            send_bytes,
            recv_bytes,
            recv_ready,
            write_log
        };
        let mut cfg = Config::new();
        cfg.async_support(true);

        let engine = Engine::new(&cfg).map_err(pyerr)?;
        let mut linker = Linker::<Ctx>::new(&engine);
        add_to_linker_async(&mut linker).map_err(pyerr)?;
        let mut root = linker.root();
        root.func_wrap_async("send-bytes", host_imports::send_bytes)
            .map_err(pyerr)?;
        root.func_wrap_async("recv-bytes", host_imports::recv_bytes)
            .map_err(pyerr)?;
        root.func_wrap("recv-ready", host_imports::recv_ready)
            .map_err(pyerr)?;
        root.func_wrap("write-log", host_imports::write_log)
            .map_err(pyerr)?;
        let wasm_path = wasm_path.unwrap_or("../env.wasm".to_string());
        let compiled_cache = wasm_compiled_cache.unwrap_or("env.wasm.compiled".to_string());

        let component =
            load_or_precompile_component(&engine, &wasm_path, &compiled_cache).map_err(pyerr)?;

        let mut wasi_builder = WasiCtxBuilder::new();
        if wasm_inherit_io {
            eprintln!("WasmRunner: Debug enabled; inheriting WASM stdio to host");
            wasi_builder.inherit_stdin();
            wasi_builder.inherit_stdout();
            wasi_builder.inherit_stderr();
        }

        let wasi = wasi_builder.build();

        let store = Store::new(
            &engine,
            Ctx {
                table: ResourceTable::new(),
                wasi,
                imports,
            },
        );

        let wasm = WasmData {
            linker: linker,
            comp: component,
            store: store,
            env: None,
            logging: runner_logging,
            id_name: id_name,
            log_tags: log_tags,
        };

        if runner_logging {
            eprintln!("WASMRunner: WasmData created");
        }

        let s = Self {
            wasm: Arc::new(Mutex::new(wasm)),
            logging: runner_logging,
        };
        Ok(s)
    }

    #[getter]
    fn running(&self) -> bool {
        return self.is_running();
    }

    fn run_msg_loop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.logging {
            eprintln!("WasmRunner: run_msg_loop()");
        }
        match self.wasm.try_lock() {
            Ok(_) => {}
            Err(_) => {
                if self.logging {
                    eprintln!("WasmRunner: run_msg_loop already running");
                }
                return Err(pyerr("WasmRunner: run_msg_loop already running"));
            }
        };
        let arc = self.wasm.clone();
        let logging = self.logging;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match arc.try_lock() {
                Ok(mut guard) => {
                    guard.instantiate().await;
                    guard.run_msg_loop().await.map_err(pyerr)
                }
                Err(_) => {
                    if logging {
                        eprintln!("WasmRunner: event_loop already running");
                    }
                    Err(pyerr("WasmRunner: event_loop already running"))
                }
            }
        })
    }

    fn close(&self) {
        if self.logging {
            eprintln!("WasmRunner: close()");
        }
    }
}

// end pymethods

#[pymodule]
fn host(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WasmRunner>()?;
    Ok(())
}

impl Drop for WasmRunner {
    fn drop(&mut self) {
        if self.logging {
            eprintln!("WasmRunner: drop()");
        }
    }
}

macro_rules! host_fn_sync_ret {
    ($fn_name:ident, $py_field:ident, ($($argn:ident : $argt:ty),*), $ret:ty) => {
        pub fn $fn_name(
            store: wasmtime::StoreContextMut<Ctx>,
            ($($argn,)*): ($($argt,)*),
        ) -> wasmtime::Result<($ret,)> {
            pyo3::Python::with_gil(|py| {
                use pyo3::types::PyAnyMethods;
                let obj = store.data().imports.$py_field.bind(py).call1(($($argn,)*))?;
                obj.extract::<$ret>()
            }).map(|v| (v,)).map_err(pyerr_to_wasmtime_err)
        }
    };
}

macro_rules! host_fn_sync_void {
    ($fn_name:ident, $py_field:ident, ($($argn:ident : $argt:ty),*)) => {
        pub fn $fn_name(
            store: wasmtime::StoreContextMut<Ctx>,
            ($($argn,)*): ($($argt,)*),
        ) -> wasmtime::Result<()> {
            pyo3::Python::with_gil(|py| {
                use pyo3::types::PyAnyMethods;
                store.data().imports.$py_field.bind(py).call1(($($argn,)*)).map(|_| ())
            }).map_err(pyerr_to_wasmtime_err)
        }
    };
}

macro_rules! host_fn_async_ret {
    ($fn_name:ident, $py_field:ident, ($($argn:ident : $argt:ty),*), $ret:ty) => {
        pub fn $fn_name(
            store: wasmtime::StoreContextMut<Ctx>,
            ($($argn,)*): ($($argt,)*),
        ) -> Box<dyn std::future::Future<Output = wasmtime::Result<($ret,)>> + Send + '_> {
            Box::new(async move {
                let fut = pyo3::Python::with_gil(|py| {
                    use pyo3::types::PyAnyMethods;
                    let coro = store.data().imports.$py_field.bind(py).call1(($($argn,)*))?;
                    pyo3_async_runtimes::tokio::into_future(coro)
                }).map_err(pyerr_to_wasmtime_err)?;
                let obj = fut.await.map_err(pyerr_to_wasmtime_err)?;
                let r = pyo3::Python::with_gil(|py| obj.extract::<$ret>(py).map_err(pyerr_to_wasmtime_err))?;
                Ok((r,))
            })
        }
    };
}

macro_rules! host_fn_async_void {
    ($fn_name:ident, $py_field:ident, ($($argn:ident : $argt:ty),*)) => {
        pub fn $fn_name(
            store: wasmtime::StoreContextMut<Ctx>,
            ($($argn,)*): ($($argt,)*),
        ) -> Box<dyn std::future::Future<Output = wasmtime::Result<()>> + Send + '_> {
            Box::new(async move {
                let fut = pyo3::Python::with_gil(|py| {
                    use pyo3::types::PyAnyMethods;
                    let coro = store.data().imports.$py_field.bind(py).call1(($($argn,)*))?;
                    pyo3_async_runtimes::tokio::into_future(coro)
                }).map_err(pyerr_to_wasmtime_err)?;

                let _obj = fut.await.map_err(pyerr_to_wasmtime_err)?;
                Ok(())
            })
        }
    };
}

mod host_imports {
    use super::{Ctx, pyerr_to_wasmtime_err};

    host_fn_async_void!(send_bytes, send_bytes, (payload: Vec<u8>));
    host_fn_async_ret!(recv_bytes, recv_bytes, (), Vec<u8>);
    host_fn_sync_ret!(recv_ready, recv_ready, (), bool);
    host_fn_sync_void!(write_log, write_log, (text: String));
}

fn load_or_precompile_component(
    engine: &Engine,
    wasm_path: &str,
    compiled_path: &str,
) -> Result<Component, String> {
    use std::fs;

    let compiled = Path::new(compiled_path);
    let wasm = Path::new(wasm_path);

    let force_recompile = std::env::var("WASMTIME_FORCE_RECOMPILE")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Decide whether to reuse the cached compiled component
    // Recompile if the compiled artifact is missing, empty, older than the wasm,
    // or if deserialization fails.
    let need_recompile = force_recompile || {
        match (fs::metadata(compiled), fs::metadata(wasm)) {
            (Ok(compiled_meta), Ok(wasm_meta)) => {
                let empty = compiled_meta.len() == 0;
                let older = match (compiled_meta.modified(), wasm_meta.modified()) {
                    (Ok(compiled_mtime), Ok(wasm_mtime)) => compiled_mtime < wasm_mtime,
                    _ => false,
                };
                !compiled.exists() || empty || older
            }
            _ => true,
        }
    };

    if !need_recompile {
        match unsafe { Component::deserialize_file(engine, compiled_path) } {
            Ok(component) => Ok(component),
            Err(_) => {
                let bytes = fs::read(wasm).map_err(|e| e.to_string())?;
                let blob = engine
                    .precompile_component(&bytes)
                    .map_err(|e| e.to_string())?;
                let _ = fs::write(compiled, blob);
                Component::from_binary(engine, &bytes).map_err(|e| e.to_string())
            }
        }
    } else {
        let bytes = fs::read(wasm).map_err(|e| e.to_string())?;
        let blob = engine
            .precompile_component(&bytes)
            .map_err(|e| e.to_string())?;
        let _ = fs::write(compiled, blob);
        Component::from_binary(engine, &bytes).map_err(|e| e.to_string())
    }
}
