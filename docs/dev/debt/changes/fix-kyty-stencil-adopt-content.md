# kyty

## add Functional

- [Stencil plane takeover](../../../core/kyty/graphics/host_gpu/renderer/cache/textureCache.cpp) (Kyty) - an image taken over as a depth target's stencil plane keeps its rendered content only when, among other conditions, it has the plane's tile mode and pitch; otherwise the plane gets the guest bytes, which may be older than that content. A plane that belonged to another live depth image is re-pointed without its stencil. Closes when: a GPU-written image with another layout is retiled into the plane, and a plane moved between depth images keeps its stencil.
