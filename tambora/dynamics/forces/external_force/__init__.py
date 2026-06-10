try:
    from .ExternalGalpyPotential import ExternalGalpyPotential
    from .LinearTideGalpyForce import LinearTideGalpyForce

except ImportError:
    class ExternalGalpyPotential:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ExternalGalpyPotential requires galpy. "
                "Refer to https://docs.galpy.org/en/stable/installation.html" \
                "for galpy installation instructions."
            )
    class LinearTideGalpyForce:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LinearTideGalpyForce requires galpy. "
                "Refer to https://docs.galpy.org/en/stable/installation.html" \
                "for galpy installation instructions."
            )