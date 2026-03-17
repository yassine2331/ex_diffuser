try:
    from datasets.Shapes3DDataset import Shapes3DDataset
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import Shapes3DDataset: {e}")
