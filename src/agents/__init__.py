__all__ = ['AINewsAgent', 'ResearchWorkflow', 'ResearchState']

def __getattr__(name):
    if name == 'AINewsAgent':
        from .main_agent import AINewsAgent
        return AINewsAgent
    elif name == 'ResearchWorkflow':
        from .graphs.research_workflow import ResearchWorkflow
        return ResearchWorkflow
    elif name == 'ResearchState':
        from .graphs.research_workflow import ResearchState
        return ResearchState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
