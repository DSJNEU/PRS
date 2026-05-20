def __getattr__(name: str):
    if name in {"app", "run_server"}:
        from prs.web import monitor

        return getattr(monitor, name)
    raise AttributeError(name)


__all__ = ["app", "run_server"]
