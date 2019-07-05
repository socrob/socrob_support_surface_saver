
class SupportSurface:
    def __init__(self, image, depth, camera_info, surface_normal):
        self.version = "0.1"
        self.image = image
        self.depth = depth
        self.camera_info = camera_info
        self.surface_normal = surface_normal
