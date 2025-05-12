import importlib

terrain_registry = dict(
    Terrain= "legged_gym.utils.terrain_parkour.terrain:Terrain_Parkour",
    BarrierTrack= "legged_gym.utils.terrain_parkour.barrier_track:BarrierTrack",
    TerrainPerlin= "legged_gym.utils.terrain_parkour.perlin:TerrainPerlin",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
