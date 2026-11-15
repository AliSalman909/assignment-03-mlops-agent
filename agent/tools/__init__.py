from agent.tools.base import ToolRegistry


def build_registry() -> ToolRegistry:
    """Create a registry with every available tool module registered."""
    from agent.tools.dvc_tools import register_dvc_tools

    registry = ToolRegistry()
    register_dvc_tools(registry)
    return registry
