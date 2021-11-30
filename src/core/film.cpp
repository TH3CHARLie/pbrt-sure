
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

// core/film.cpp*
#include "film.h"

#include "imageio.h"
#include "paramset.h"
#include "stats.h"
#include <limits>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Film pixels", filmPixelMemory);

// Film Method Definitions
Film::Film(const Point2i &resolution, const Bounds2f &cropWindow,
           std::unique_ptr<Filter> filt, Float diagonal,
           const std::string &filename, Float scale, Float maxSampleLuminance)
    : fullResolution(resolution),
      diagonal(diagonal * .001),
      filter(std::move(filt)),
      filename(filename),
      scale(scale),
      maxSampleLuminance(maxSampleLuminance) {
    // Compute film image bounds
    croppedPixelBounds =
        Bounds2i(Point2i(std::ceil(fullResolution.x * cropWindow.pMin.x),
                         std::ceil(fullResolution.y * cropWindow.pMin.y)),
                 Point2i(std::ceil(fullResolution.x * cropWindow.pMax.x),
                         std::ceil(fullResolution.y * cropWindow.pMax.y)));
    LOG(INFO) << "Created film with full resolution " << resolution
              << ". Crop window of " << cropWindow << " -> croppedPixelBounds "
              << croppedPixelBounds;

    // Allocate film image storage
    pixels = std::unique_ptr<Pixel[]>(new Pixel[croppedPixelBounds.Area()]);
    filmPixelMemory += croppedPixelBounds.Area() * sizeof(Pixel);

    // Precompute filter weight table
    int offset = 0;
    for (int y = 0; y < filterTableWidth; ++y) {
        for (int x = 0; x < filterTableWidth; ++x, ++offset) {
            Point2f p;
            p.x = (x + 0.5f) * filter->radius.x / filterTableWidth;
            p.y = (y + 0.5f) * filter->radius.y / filterTableWidth;
            filterTable[offset] = filter->Evaluate(p);
        }
    }
}

Bounds2i Film::GetSampleBounds() const {
    Bounds2f floatBounds(Floor(Point2f(croppedPixelBounds.pMin) +
                               Vector2f(0.5f, 0.5f) - filter->radius),
                         Ceil(Point2f(croppedPixelBounds.pMax) -
                              Vector2f(0.5f, 0.5f) + filter->radius));
    return (Bounds2i)floatBounds;
}

Bounds2f Film::GetPhysicalExtent() const {
    Float aspect = (Float)fullResolution.y / (Float)fullResolution.x;
    Float x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
    Float y = aspect * x;
    return Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));
}

std::unique_ptr<FilmTile> Film::GetFilmTile(const Bounds2i &sampleBounds) {
    // Bound image pixels that samples in _sampleBounds_ contribute to
    Vector2f halfPixel = Vector2f(0.5f, 0.5f);
    Bounds2f floatBounds = (Bounds2f)sampleBounds;
    Point2i p0 = (Point2i)Ceil(floatBounds.pMin - halfPixel - filter->radius);
    Point2i p1 = (Point2i)Floor(floatBounds.pMax - halfPixel + filter->radius) +
                 Point2i(1, 1);
    Bounds2i tilePixelBounds = Intersect(Bounds2i(p0, p1), croppedPixelBounds);
    return std::unique_ptr<FilmTile>(
        new FilmTile(tilePixelBounds, filter->radius, filterTable,
                     filterTableWidth, maxSampleLuminance));
}

void Film::Clear() {
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        for (int c = 0; c < 3; ++c) pixel.splatXYZ[c] = pixel.xyz[c] = 0;
        pixel.filterWeightSum = 0;
    }
}

void Film::MergeFilmTile(std::unique_ptr<FilmTile> tile) {
    ProfilePhase p(Prof::MergeFilmTile);
    VLOG(1) << "Merging film tile " << tile->pixelBounds;
    std::lock_guard<std::mutex> lock(mutex);
    for (Point2i pixel : tile->GetPixelBounds()) {
        // Merge _pixel_ into _Film::pixels_
        const FilmTilePixel &tilePixel = tile->GetPixel(pixel);
        Pixel &mergePixel = GetPixel(pixel);
        Float xyz[3];
        tilePixel.contribSum.ToXYZ(xyz);
        for (int i = 0; i < 3; ++i) {
            mergePixel.xyz[i] += xyz[i];
            mergePixel.color_mean[i] += xyz[i];
            mergePixel.color_squared_sum[i] += (xyz[i] * xyz[i]);
        }
        mergePixel.filterWeightSum += tilePixel.filterWeightSum;
        tilePixel.normal.ToXYZ(xyz);
        for (int i = 0; i < 3; ++i) {
            mergePixel.normal_mean[i] += xyz[i];
            mergePixel.normal_squared_sum[i] += (xyz[i] * xyz[i]);
        }
        tilePixel.texture_value.ToXYZ(xyz);
        for (int i = 0; i < 3; ++i) {
            mergePixel.texture_mean[i] += xyz[i];
            mergePixel.texture_squared_sum[i] += (xyz[i] * xyz[i]);
        }
        mergePixel.depth_mean += tilePixel.depth;
        mergePixel.depth_squared_sum += (tilePixel.depth * tilePixel.depth);
    }
}

void Film::SetImage(const Spectrum *img) const {
    int nPixels = croppedPixelBounds.Area();
    for (int i = 0; i < nPixels; ++i) {
        Pixel &p = pixels[i];
        img[i].ToXYZ(p.xyz);
        p.filterWeightSum = 1;
        p.splatXYZ[0] = p.splatXYZ[1] = p.splatXYZ[2] = 0;
    }
}

void Film::AddSplat(const Point2f &p, Spectrum v) {
    ProfilePhase pp(Prof::SplatFilm);

    if (v.HasNaNs()) {
        LOG(ERROR) << StringPrintf(
            "Ignoring splatted spectrum with NaN values "
            "at (%f, %f)",
            p.x, p.y);
        return;
    } else if (v.y() < 0.) {
        LOG(ERROR) << StringPrintf(
            "Ignoring splatted spectrum with negative "
            "luminance %f at (%f, %f)",
            v.y(), p.x, p.y);
        return;
    } else if (std::isinf(v.y())) {
        LOG(ERROR) << StringPrintf(
            "Ignoring splatted spectrum with infinite "
            "luminance at (%f, %f)",
            p.x, p.y);
        return;
    }

    Point2i pi = Point2i(Floor(p));
    if (!InsideExclusive(pi, croppedPixelBounds)) return;
    if (v.y() > maxSampleLuminance) v *= maxSampleLuminance / v.y();
    Float xyz[3];
    v.ToXYZ(xyz);
    Pixel &pixel = GetPixel(pi);
    for (int i = 0; i < 3; ++i) pixel.splatXYZ[i].Add(xyz[i]);
}

void Film::WriteImage(Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG(INFO)
        << "Converting image to RGB and computing final weighted pixel values";
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    Float sample_cnt = 0;
    int pixel_cnt = 0;
    for (Point2i p : croppedPixelBounds) {
        // Convert pixel XYZ color to RGB
        Pixel &pixel = GetPixel(p);
        XYZToRGB(pixel.xyz, &rgb[3 * offset]);

        // Normalize pixel with weight sum
        Float filterWeightSum = pixel.filterWeightSum;
        if (filterWeightSum != 0) {
            Float invWt = (Float)1 / filterWeightSum;
            rgb[3 * offset] = std::max((Float)0, rgb[3 * offset] * invWt);
            rgb[3 * offset + 1] =
                std::max((Float)0, rgb[3 * offset + 1] * invWt);
            rgb[3 * offset + 2] =
                std::max((Float)0, rgb[3 * offset + 2] * invWt);
        }

        // Add splat value at pixel
        Float splatRGB[3];
        Float splatXYZ[3] = {pixel.splatXYZ[0], pixel.splatXYZ[1],
                             pixel.splatXYZ[2]};
        XYZToRGB(splatXYZ, splatRGB);
        rgb[3 * offset] += splatScale * splatRGB[0];
        rgb[3 * offset + 1] += splatScale * splatRGB[1];
        rgb[3 * offset + 2] += splatScale * splatRGB[2];

        // Scale pixel value by _scale_
        rgb[3 * offset] *= scale;
        rgb[3 * offset + 1] *= scale;
        rgb[3 * offset + 2] *= scale;
        ++offset;
        pixel_cnt++;
        sample_cnt += pixel.filterWeightSum;
    }

    // Write RGB image
    LOG(INFO) << "Writing image " << filename << " with bounds "
              << croppedPixelBounds;
    pbrt::WriteImage(filename, &rgb[0], croppedPixelBounds, fullResolution);
    std::cout << "average samples: " << (1.0 * sample_cnt) / pixel_cnt << '\n';
}

void Film::WriteColorImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    std::unique_ptr<Float[]> var_rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        XYZToRGB(pixel.color_mean, &rgb[3 * offset]);
        XYZToRGB(pixel.color_variance, &var_rgb[3 * offset]);
        ++offset;
    }
    pbrt::WriteImage("sure_color_mean.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
    pbrt::WriteImage("sure_color_variance.png", &var_rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::WriteTextureImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    std::unique_ptr<Float[]> var_rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        XYZToRGB(pixel.texture_mean, &rgb[3 * offset]);
        XYZToRGB(pixel.texture_variance, &var_rgb[3 * offset]);
        ++offset;
    }
    pbrt::WriteImage("sure_texture_mean.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
    pbrt::WriteImage("sure_texture_variance.png", &var_rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::WriteNormalImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    std::unique_ptr<Float[]> var_rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        XYZToRGB(pixel.normal_mean, &rgb[3 * offset]);
        XYZToRGB(pixel.normal_variance, &var_rgb[3 * offset]);
        ++offset;
    }
    pbrt::WriteImage("sure_normal_mean.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
    pbrt::WriteImage("sure_normal_variance.png", &var_rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::WriteDepthImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    std::unique_ptr<Float[]> var_rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        // Normalize pixel with weight sum
        rgb[3 * offset] = rgb[3 * offset + 1] = rgb[3 * offset + 2] =
                std::max((Float)0, pixel.depth_mean);
        var_rgb[3 * offset] = var_rgb[3 * offset + 1] = var_rgb[3 * offset + 2] =
                std::max((Float)0, pixel.depth_variance);
        ++offset;
    }
    pbrt::WriteImage("sure_depth_mean.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
    pbrt::WriteImage("sure_depth_variance.png", &var_rgb[0], croppedPixelBounds,
                     fullResolution);
}


void Film::WriteSamplingDensityImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        // Normalize pixel with weight sum
        rgb[3 * offset] = rgb[3 * offset + 1] = rgb[3 * offset + 2] =
                std::max((Float)0, pixel.density);
        ++offset;
    }
    pbrt::WriteImage("sure_sampling_density.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::WriteFilteredImage(const std::string& filename) {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        XYZToRGB(pixel.best_filtered_color, &rgb[3 * offset]);
        ++offset;
    }
    pbrt::WriteImage(filename, &rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::WriteSUREEstimatedErrorImage() {
    std::unique_ptr<Float[]> rgb(new Float[3 * croppedPixelBounds.Area()]);
    int offset = 0;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        Float avg = pixel.best_mse;
        rgb[3 * offset] = rgb[3 * offset + 1] = rgb[3 * offset + 2] = std::max((Float)0, avg);
        ++offset;
    }
    pbrt::WriteImage("sure_error.png", &rgb[0], croppedPixelBounds,
                     fullResolution);
}

void Film::Preprocess_SURE_ext() {
    Float max_depth = -1.0f;
    for (Point2i p : croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        // Normalize pixel with weight sum
        Float filterWeightSum = pixel.filterWeightSum;
        if (filterWeightSum != 0) {
            Float invWt = (Float)1 / filterWeightSum;
            max_depth = std::max(max_depth, pixel.depth_mean * invWt);
        }
    }
    for (Point2i p: croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        // Normalize pixel with weight sum
        Float filterWeightSum = pixel.filterWeightSum;
        if (filterWeightSum != 0) {
            Float invWt = (Float)1 / filterWeightSum;
            Float invWt1 = (Float)1 / (filterWeightSum - 1);
            for (int i = 0; i < 3; ++i) {
                pixel.color_variance[i] = (pixel.color_squared_sum[i] - pixel.color_mean[i] * pixel.color_mean[i] * invWt) * invWt1;
                pixel.normal_variance[i] = (pixel.normal_squared_sum[i] - pixel.normal_mean[i] * pixel.normal_mean[i] * invWt) * invWt1;
                pixel.texture_variance[i] = (pixel.texture_squared_sum[i] - pixel.texture_mean[i] * pixel.texture_mean[i] * invWt) * invWt1;
                pixel.color_mean[i] = pixel.color_mean[i] * invWt;
                pixel.normal_mean[i] = pixel.normal_mean[i] * invWt;
                pixel.texture_mean[i] = pixel.texture_mean[i] * invWt;
            }
            pixel.depth_mean = pixel.depth_mean / max_depth;
            pixel.depth_variance = (pixel.depth_squared_sum / (max_depth * max_depth) -
                                    pixel.depth_mean * pixel.depth_mean * invWt) * invWt1;
            pixel.depth_mean *= invWt;
        }
    }
}

void Film::CrossBilateralFilter(Float sigma_S_array[], Float sigma_R, Float sigma_T, Float sigma_N, Float sigma_D) {
    for (int i = 0; i < BANK_SIZE; ++i) {
        Float sigma_S = sigma_S_array[i];
        int radius = (int)round(sigma_S * 2);
        for (int yy = croppedPixelBounds.pMin.y; yy < croppedPixelBounds.pMax.y; ++yy) {
            for (int xx = croppedPixelBounds.pMin.x; xx < croppedPixelBounds.pMax.x; ++xx) {
                int xl = std::max(xx - radius, croppedPixelBounds.pMin.x);
                int xu = std::min(xx + radius, croppedPixelBounds.pMax.x - 1);
                int yl = std::max(yy - radius, croppedPixelBounds.pMin.y);
                int yu = std::min(yy + radius, croppedPixelBounds.pMax.y - 1);
                Pixel& pixel = GetPixel(Point2i(xx, yy));
                Vector3f texture_vec(pixel.texture_mean[0], pixel.texture_mean[1], pixel.texture_mean[2]);
                Vector3f normal_vec(pixel.normal_mean[0], pixel.normal_mean[1], pixel.normal_mean[2]);
                for (int c = 0; c < 3; ++c) {
                    Float sum_weight = 0.0;
                    Float sum_weighted_color = 0.0;
                    Float sum_weighted_color_squared = 0.0;
                    if (sigma_S <= 0.0) {
                        sum_weight = 1.0;
                        sum_weighted_color = 1.0 * pixel.color_mean[c];
                        sum_weighted_color_squared = 1.0 * pixel.color_mean[c] * pixel.color_mean[c];
                    }
                    for (int y = yl; y < yu; ++y) {
                        Float y_distance = (y - yy) * (y - yy);
                        for (int x = xl; x < xu; ++x) {
                            Float x_distance = (x - xx) * (x - xx);
                            Float spatial_distance = x_distance + y_distance;
                            Pixel &cur_pixel = GetPixel(Point2i(x, y)); 
                            Float color_distance = (pixel.color_mean[c] - cur_pixel.color_mean[c]) * (pixel.color_mean[c] - cur_pixel.color_mean[c]);
                            Float spatial_term = sigma_S <= 0 ? 0.0 : -spatial_distance / (2 * sigma_S * sigma_S);
                            Float color_term = -color_distance / (2 * sigma_R * sigma_R);
                            Vector3f cur_texture_vec(cur_pixel.texture_mean[0], cur_pixel.texture_mean[1], cur_pixel.texture_mean[2]);
                            Vector3f cur_normal_vec(cur_pixel.normal_mean[0], cur_pixel.normal_mean[1], cur_pixel.normal_mean[2]);
                            Float texture_term = -(texture_vec - cur_texture_vec).LengthSquared() / (2 * sigma_T * sigma_T);
                            Float normal_term = -(normal_vec - cur_normal_vec).LengthSquared() / (2 * sigma_N * sigma_N);
                            Float depth_term = -((pixel.depth_mean - cur_pixel.depth_mean) * (pixel.depth_mean - cur_pixel.depth_mean)) / (2 * sigma_D * sigma_D);
                            // TODO: this is following Tzu-Mao's implementation
                            // color_term = 0;
                            Float w = std::exp(spatial_term + color_term + texture_term + normal_term + depth_term);
                            sum_weight += w;
                            sum_weighted_color += (w * cur_pixel.color_mean[c]);
                            sum_weighted_color_squared += (w * cur_pixel.color_mean[c] * cur_pixel.color_mean[c]);
                        }
                    }
                    pixel.filtered_color[c + 3 * i] = sum_weighted_color / sum_weight;
                    Float dFy = 1.0 / sum_weight + 1.0 / (sigma_R * sigma_R) * (sum_weighted_color_squared / sum_weight - pixel.filtered_color[c] * pixel.filtered_color[c]);
                    Float sure_estimated_error = (pixel.filtered_color[c] - pixel.color_mean[c]) * (pixel.filtered_color[c] - pixel.color_mean[c]) + pixel.color_variance[c] * (2 * dFy - 1.0);
                    pixel.mse_estimation[c + 3 * i] = sure_estimated_error;
                }
                Float error_sum = 0;
                for (int c = 0; c < 3; ++c) {
                    error_sum += pixel.mse_estimation[c + 3 * i];
                }
                error_sum = std::max((Float)0.0, error_sum);
                pixel.avg_mse[i] = error_sum / 3.0;
            }
        }
    }
    for (int yy = croppedPixelBounds.pMin.y; yy < croppedPixelBounds.pMax.y; ++yy) {
        for (int xx = croppedPixelBounds.pMin.x; xx < croppedPixelBounds.pMax.x; ++xx) {
            Pixel& pixel = GetPixel(Point2i(xx, yy));
            Float min_error = std::numeric_limits<Float>::infinity();
            for (int i = 0; i < BANK_SIZE; ++i) {
                if (pixel.avg_mse[i] < min_error) {
                    pixel.best_mse = pixel.avg_mse[i];
                    for (int c = 0; c < 3; ++c) {
                        pixel.best_filtered_color[c] = pixel.filtered_color[c + 3 * i];
                    }
                    min_error = pixel.avg_mse[i];
                }
            }
        }
    }
}

void Film::UpdateSampleLimit(int totalSampleBudget, int maxPerPixelBudget) {

    Float total_density = 0;
    for (Point2i p: croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        Float squared_luminance = pixel.best_filtered_color[0] * pixel.best_filtered_color[0] + 
                                  pixel.best_filtered_color[1] * pixel.best_filtered_color[1] + 
                                  pixel.best_filtered_color[2] * pixel.best_filtered_color[2];
        pixel.density = (pixel.best_mse + 0.0) / (squared_luminance + 1e-9);
        total_density += pixel.density;
    }

    for (Point2i p: croppedPixelBounds) {
        Pixel &pixel = GetPixel(p);
        pixel.sample_limit = std::min((int)ceil(pixel.density / total_density * totalSampleBudget), maxPerPixelBudget);
        pixel.sample_limit = std::max(11, pixel.sample_limit);
    }
}

int Film::GetSampleLimit(const Point2i& p) {
    if (InsideExclusive(p, croppedPixelBounds)) {
        Pixel& pixel = GetPixel(p);
        return pixel.sample_limit;    
    }
    return 0;
}

Film *CreateFilm(const ParamSet &params, std::unique_ptr<Filter> filter) {
    std::string filename;
    if (PbrtOptions.imageFile != "") {
        filename = PbrtOptions.imageFile;
        std::string paramsFilename = params.FindOneString("filename", "");
        if (paramsFilename != "")
            Warning(
                "Output filename supplied on command line, \"%s\" is "
                "overriding "
                "filename provided in scene description file, \"%s\".",
                PbrtOptions.imageFile.c_str(), paramsFilename.c_str());
    } else
        filename = params.FindOneString("filename", "pbrt.exr");

    int xres = params.FindOneInt("xresolution", 1280);
    int yres = params.FindOneInt("yresolution", 720);
    if (PbrtOptions.quickRender) xres = std::max(1, xres / 4);
    if (PbrtOptions.quickRender) yres = std::max(1, yres / 4);
    Bounds2f crop;
    int cwi;
    const Float *cr = params.FindFloat("cropwindow", &cwi);
    if (cr && cwi == 4) {
        crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
        crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
        crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
        crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);
    } else if (cr)
        Error("%d values supplied for \"cropwindow\". Expected 4.", cwi);
    else
        crop = Bounds2f(Point2f(Clamp(PbrtOptions.cropWindow[0][0], 0, 1),
                                Clamp(PbrtOptions.cropWindow[1][0], 0, 1)),
                        Point2f(Clamp(PbrtOptions.cropWindow[0][1], 0, 1),
                                Clamp(PbrtOptions.cropWindow[1][1], 0, 1)));

    Float scale = params.FindOneFloat("scale", 1.);
    Float diagonal = params.FindOneFloat("diagonal", 35.);
    Float maxSampleLuminance =
        params.FindOneFloat("maxsampleluminance", Infinity);
    return new Film(Point2i(xres, yres), crop, std::move(filter), diagonal,
                    filename, scale, maxSampleLuminance);
}

}  // namespace pbrt
