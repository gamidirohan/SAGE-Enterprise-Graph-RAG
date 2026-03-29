import importlib
import sys
import types


class _SessionState(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Empty:
    def container(self):
        return _Ctx()


def _build_fake_streamlit():
    module = types.ModuleType("streamlit")
    module.session_state = _SessionState()
    module.cache_resource = lambda f: f
    module.set_page_config = lambda **_kwargs: None
    module.title = lambda *_args, **_kwargs: None
    module.subheader = lambda *_args, **_kwargs: None
    module.markdown = lambda *_args, **_kwargs: None
    module.checkbox = lambda *_args, **_kwargs: False
    module.spinner = lambda *_args, **_kwargs: _Ctx()
    module.columns = lambda n: tuple(_Ctx() for _ in range(n if isinstance(n, int) else len(n)))
    module.table = lambda *_args, **_kwargs: None
    module.warning = lambda *_args, **_kwargs: None
    module.success = lambda *_args, **_kwargs: None
    module.error = lambda *_args, **_kwargs: None
    module.empty = lambda: _Empty()
    module.text_input = lambda *_args, **_kwargs: ""
    module.write = lambda *_args, **_kwargs: None
    module.button = lambda *_args, **_kwargs: False
    module.rerun = lambda: None
    return module


def _import_graph_rag(monkeypatch):
    fake_streamlit = _build_fake_streamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    if "app.graph_rag" in sys.modules:
        del sys.modules["app.graph_rag"]
    module = importlib.import_module("app.graph_rag")
    return module, fake_streamlit


def test_open_neo4j_session_uses_database(monkeypatch):
    module, _st = _import_graph_rag(monkeypatch)

    class Driver:
        def __init__(self):
            self.kwargs = None

        def session(self, **kwargs):
            self.kwargs = kwargs
            return object()

    driver = Driver()
    module.utils.open_neo4j_session(driver, "neo_db")
    assert driver.kwargs == {"database": "neo_db"}


def test_generate_groq_response_with_no_documents(monkeypatch):
    module, _st = _import_graph_rag(monkeypatch)
    result = module.generate_groq_response("q", [])
    assert result == "No relevant information found."


def test_generate_groq_response_success(monkeypatch):
    module, _st = _import_graph_rag(monkeypatch)
    monkeypatch.setattr(module.services, "generate_streamlit_response", lambda _q, _d: "final")

    result = module.generate_groq_response("q", ["Chunk Summary: ctx, Document: d"])
    assert result == "final"


def test_on_enter_updates_session_state(monkeypatch):
    module, st = _import_graph_rag(monkeypatch)
    st.session_state.user_input = "hello"
    st.session_state.processing = False

    module.on_enter()

    assert st.session_state.temp_input == "hello"
    assert st.session_state.user_input == ""
    assert st.session_state.processing is True
