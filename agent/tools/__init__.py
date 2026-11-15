from agent.tools.base import ToolRegistry


def build_registry() -> ToolRegistry:
    """Create a registry with every available tool module registered."""
    from agent.tools.airflow_tools import register_airflow_tools
    from agent.tools.dvc_tools import register_dvc_tools
    from agent.tools.mlflow_tools import register_mlflow_tools

    registry = ToolRegistry()
    register_dvc_tools(registry)
    register_mlflow_tools(registry)
    register_airflow_tools(registry)
    return registry
