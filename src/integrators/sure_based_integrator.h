#ifndef SURE_BASED_INTEGRATOR_H
#define SURE_BASED_INTEGRATOR_H

#include <cstdint>
#include <memory>

#include "camera.h"
#include "integrator.h"
#include "integrators/path.h"
#include "paramset.h"
#include "pbrt.h"
#include "sampler.h"

namespace pbrt {

class SUREBasedIntegrator : public Integrator {
  public:
    SUREBasedIntegrator(std::shared_ptr<Sampler> sampler,
                        std::shared_ptr<const Camera> camera,
                        std::shared_ptr<PathIntegrator> path_integrator,
                        const Bounds2i pixel_bounds,
                        const int64_t sample_budget,
                        const int64_t num_initial_samples)
        : sampler(sampler),
          camera(camera),
          path_integrator(path_integrator),
          pixel_bounds(pixel_bounds),
          sample_budget(sample_budget),
          num_initial_samples(num_initial_samples) {}

    virtual ~SUREBasedIntegrator() {}

    virtual void Render(const Scene &scene);

  private:
    std::shared_ptr<Sampler> sampler;
    std::shared_ptr<const Camera> camera;
    std::shared_ptr<PathIntegrator> path_integrator;
    const Bounds2i pixel_bounds;
    const int64_t sample_budget;
    const int64_t num_initial_samples;
};

SUREBasedIntegrator *CreateSUREBasedIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);
}  // namespace pbrt

#endif
