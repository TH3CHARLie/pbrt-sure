
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_FILM_H
#define PBRT_CORE_FILM_H

// core/film.h*
#include "filter.h"
#include "geometry.h"
#include "parallel.h"
#include "pbrt.h"
#include "spectrum.h"
#include "stats.h"
#include "sure_based_utility.h"

namespace pbrt {

// FilmTilePixel Declarations
struct FilmTilePixel {
    Spectrum contribSum = 0.f;
    Float filterWeightSum = 0.f;
    Spectrum normal = 0.f;
    Spectrum texture_value = 0.f;
    Float depth = 0.f;
};

constexpr int BANK_SIZE = 5;
// Film Declarations
class Film {
  public:
    // Film Public Methods
    Film(const Point2i &resolution, const Bounds2f &cropWindow,
         std::unique_ptr<Filter> filter, Float diagonal,
         const std::string &filename, Float scale,
         Float maxSampleLuminance = Infinity);
    Bounds2i GetSampleBounds() const;
    Bounds2f GetPhysicalExtent() const;
    std::unique_ptr<FilmTile> GetFilmTile(const Bounds2i &sampleBounds);
    void MergeFilmTile(std::unique_ptr<FilmTile> tile);
    void SetImage(const Spectrum *img) const;
    void AddSplat(const Point2f &p, Spectrum v);
    void WriteImage(Float splatScale = 1);
    void WriteColorImage();
    void WriteTextureImage();
    void WriteNormalImage();
    void WriteDepthImage();
    void WriteSamplingDensityImage();
    void WriteFilteredImage(const std::string &filename);
    void WriteSUREEstimatedErrorImage();
    void Preprocess_SURE_ext();
    void CrossBilateralFilter(Float sigma_S_array[], Float sigma_R,
                              Float sigma_T, Float sigma_N, Float sigma_D);
    void UpdateSampleLimit(int totalSampleBudget, int maxPerPixelBudget);
    void Clear();
    int GetSampleLimit(const Point2i &pixel);

    // Film Public Data
    const Point2i fullResolution;
    const Float diagonal;
    std::unique_ptr<Filter> filter;
    const std::string filename;
    Bounds2i croppedPixelBounds;

  private:
    // Film Private Data
    struct Pixel {
        Pixel() {
            xyz[0] = xyz[1] = xyz[2] = filterWeightSum = 0;
            color_mean[0] = color_mean[1] = color_mean[2] = 0;
            color_variance[0] = color_variance[1] = color_variance[2] = 0;
            normal_mean[0] = normal_mean[1] = normal_mean[2] = 0;
            normal_variance[0] = normal_variance[1] = normal_variance[2] = 0;
            texture_mean[0] = texture_mean[1] = texture_mean[2] = 0;
            texture_variance[0] = texture_variance[1] = texture_variance[2] = 0;
            depth_mean = 0;
            depth_variance = 0;
            filtered_color[0] = filtered_color[1] = filtered_color[2] = 0;
            mse_estimation[0] = mse_estimation[1] = mse_estimation[2] = 0;
            sample_limit = 0;
        }

        Float xyz[3];
        Float filterWeightSum;
        AtomicFloat splatXYZ[3];
        Float pad;
        Float color_mean[3];
        Float color_variance[3];
        Float normal_mean[3];
        Float normal_variance[3];
        Float texture_mean[3];
        Float texture_variance[3];
        Float depth_mean;
        Float depth_variance;
        Float filtered_color[3 * BANK_SIZE];
        Float mse_estimation[3 * BANK_SIZE];
        Float avg_mse[BANK_SIZE];
        Float best_mse;
        Float best_filtered_color[3];
        int sample_limit;

        Float density;
    };
    std::unique_ptr<Pixel[]> pixels;
    static PBRT_CONSTEXPR int filterTableWidth = 16;
    Float filterTable[filterTableWidth * filterTableWidth];
    std::mutex mutex;
    const Float scale;
    const Float maxSampleLuminance;

    // Film Private Methods
    Pixel &GetPixel(const Point2i &p) {
        CHECK(InsideExclusive(p, croppedPixelBounds));
        int width = croppedPixelBounds.pMax.x - croppedPixelBounds.pMin.x;
        int offset = (p.x - croppedPixelBounds.pMin.x) +
                     (p.y - croppedPixelBounds.pMin.y) * width;
        return pixels[offset];
    }
};

class FilmTile {
  public:
    // FilmTile Public Methods
    FilmTile(const Bounds2i &pixelBounds, const Vector2f &filterRadius,
             const Float *filterTable, int filterTableSize,
             Float maxSampleLuminance)
        : pixelBounds(pixelBounds),
          filterRadius(filterRadius),
          invFilterRadius(1 / filterRadius.x, 1 / filterRadius.y),
          filterTable(filterTable),
          filterTableSize(filterTableSize),
          maxSampleLuminance(maxSampleLuminance) {
        pixels = std::vector<FilmTilePixel>(std::max(0, pixelBounds.Area()));
    }
    void AddSample(const Point2f &pFilm, Spectrum L, Float sampleWeight = 1.) {
        ProfilePhase _(Prof::AddFilmSample);
        if (L.y() > maxSampleLuminance) L *= maxSampleLuminance / L.y();
        // Compute sample's raster bounds
        Point2f pFilmDiscrete = pFilm - Vector2f(0.5f, 0.5f);
        Point2i p0 = (Point2i)Ceil(pFilmDiscrete - filterRadius);
        Point2i p1 =
            (Point2i)Floor(pFilmDiscrete + filterRadius) + Point2i(1, 1);
        p0 = Max(p0, pixelBounds.pMin);
        p1 = Min(p1, pixelBounds.pMax);

        // Loop over filter support and add sample to pixel arrays

        // Precompute $x$ and $y$ filter table offsets
        int *ifx = ALLOCA(int, p1.x - p0.x);
        for (int x = p0.x; x < p1.x; ++x) {
            Float fx = std::abs((x - pFilmDiscrete.x) * invFilterRadius.x *
                                filterTableSize);
            ifx[x - p0.x] = std::min((int)std::floor(fx), filterTableSize - 1);
        }
        int *ify = ALLOCA(int, p1.y - p0.y);
        for (int y = p0.y; y < p1.y; ++y) {
            Float fy = std::abs((y - pFilmDiscrete.y) * invFilterRadius.y *
                                filterTableSize);
            ify[y - p0.y] = std::min((int)std::floor(fy), filterTableSize - 1);
        }
        for (int y = p0.y; y < p1.y; ++y) {
            for (int x = p0.x; x < p1.x; ++x) {
                // Evaluate filter value at $(x,y)$ pixel
                int offset = ify[y - p0.y] * filterTableSize + ifx[x - p0.x];
                Float filterWeight = filterTable[offset];

                // Update pixel values with filtered sample contribution
                FilmTilePixel &pixel = GetPixel(Point2i(x, y));
                pixel.contribSum += L * sampleWeight * filterWeight;
                pixel.filterWeightSum += filterWeight;
            }
        }
    }
    void AddSample_SURE_ext(const Point2f &pFilm, Spectrum L,
                            const SUREBasedAuxiliaryData &auxiliary,
                            Float sampleWeight = 1.) {
        ProfilePhase _(Prof::AddFilmSample);
        if (L.y() > maxSampleLuminance) L *= maxSampleLuminance / L.y();
        // Compute sample's raster bounds
        Point2f pFilmDiscrete = pFilm - Vector2f(0.5f, 0.5f);
        Point2i p0 = (Point2i)Ceil(pFilmDiscrete - filterRadius);
        Point2i p1 =
            (Point2i)Floor(pFilmDiscrete + filterRadius) + Point2i(1, 1);
        p0 = Max(p0, pixelBounds.pMin);
        p1 = Min(p1, pixelBounds.pMax);

        // Loop over filter support and add sample to pixel arrays

        // Precompute $x$ and $y$ filter table offsets
        int *ifx = ALLOCA(int, p1.x - p0.x);
        for (int x = p0.x; x < p1.x; ++x) {
            Float fx = std::abs((x - pFilmDiscrete.x) * invFilterRadius.x *
                                filterTableSize);
            ifx[x - p0.x] = std::min((int)std::floor(fx), filterTableSize - 1);
        }
        int *ify = ALLOCA(int, p1.y - p0.y);
        for (int y = p0.y; y < p1.y; ++y) {
            Float fy = std::abs((y - pFilmDiscrete.y) * invFilterRadius.y *
                                filterTableSize);
            ify[y - p0.y] = std::min((int)std::floor(fy), filterTableSize - 1);
        }
        for (int y = p0.y; y < p1.y; ++y) {
            for (int x = p0.x; x < p1.x; ++x) {
                // Evaluate filter value at $(x,y)$ pixel
                int offset = ify[y - p0.y] * filterTableSize + ifx[x - p0.x];
                Float filterWeight = filterTable[offset];

                // Update pixel values with filtered sample contribution
                FilmTilePixel &pixel = GetPixel(Point2i(x, y));
                pixel.contribSum += L * sampleWeight * filterWeight;
                pixel.filterWeightSum += filterWeight;
                for (int ii = 0; ii < 3; ++ii) {
                    pixel.normal[ii] +=
                        auxiliary.normal[ii] * sampleWeight * filterWeight;
                }
                pixel.texture_value +=
                    auxiliary.texture_value * sampleWeight * filterWeight;
                pixel.depth += auxiliary.depth;
            }
        }
    }
    FilmTilePixel &GetPixel(const Point2i &p) {
        CHECK(InsideExclusive(p, pixelBounds));
        int width = pixelBounds.pMax.x - pixelBounds.pMin.x;
        int offset =
            (p.x - pixelBounds.pMin.x) + (p.y - pixelBounds.pMin.y) * width;
        return pixels[offset];
    }
    const FilmTilePixel &GetPixel(const Point2i &p) const {
        CHECK(InsideExclusive(p, pixelBounds));
        int width = pixelBounds.pMax.x - pixelBounds.pMin.x;
        int offset =
            (p.x - pixelBounds.pMin.x) + (p.y - pixelBounds.pMin.y) * width;
        return pixels[offset];
    }
    Bounds2i GetPixelBounds() const { return pixelBounds; }

  private:
    // FilmTile Private Data
    const Bounds2i pixelBounds;
    const Vector2f filterRadius, invFilterRadius;
    const Float *filterTable;
    const int filterTableSize;
    std::vector<FilmTilePixel> pixels;
    const Float maxSampleLuminance;
    friend class Film;
};

Film *CreateFilm(const ParamSet &params, std::unique_ptr<Filter> filter);

}  // namespace pbrt

#endif  // PBRT_CORE_FILM_H
