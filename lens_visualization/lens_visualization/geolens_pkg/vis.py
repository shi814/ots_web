# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Visualization functions for GeoLens.

Functions:
    Ray Sampling (2D):
        - sample_parallel_2D(): Sample parallel rays (2D) in object space
        - sample_point_source_2D(): Sample point source rays (2D) in object space

    2D Layout Visualization:
        - draw_layout(): Plot 2D lens layout with ray tracing
        - draw_lens_2d(): Draw lens layout in a 2D plot
        - draw_ray_2d(): Plot ray paths

    3D Layout Visualization:
        - draw_layout_3d(): Draw 3D layout of the lens system

    3D Barrier Generation:
        - create_barrier(): Create a 3D barrier for the lens system
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..basics import DEFAULT_WAVE, DEPTH, WAVE_RGB
from ..optics.ray import Ray


class GeoLensVis:
    # ====================================================================================
    # Ray sampling functions for 2D layout
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(
        self,
        fov=0.0,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        entrance_pupil=True,
        depth=0.0,
    ):
        """Sample parallel rays (2D) in object space.

        Used for (1) drawing lens setup, (2) 2D geometric optics calculation, for example, refocusing to infinity

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to 0.0.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            plane (str, optional): sampling plane. Defaults to "meridional" (y-z plane).
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to True.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        # Sample ray origins, shape [num_rays, 3]
        if plane == "sagittal":
            ray_o = torch.stack(
                (
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), 0),
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_o = torch.stack(
                (
                    torch.full((num_rays,), 0),
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Sample ray directions, shape [num_rays, 3]
        if plane == "sagittal":
            ray_d = torch.stack(
                (
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_d = torch.stack(
                (
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Form rays and propagate to the target depth
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self,
        fov=0.0,
        depth=DEPTH,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
    ):
        """Sample point source rays (2D) in object space.

        Used for (1) drawing lens setup.

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to DEPTH.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample point on the object plane
        ray_o = torch.tensor([depth * float(np.tan(np.deg2rad(fov))), 0.0, depth])
        ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)

        # Sample points (second point) on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        x2 = torch.linspace(-pupilr, pupilr, num_rays) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        ray_o2 = torch.stack((x2, y2, z2), axis=1)

        # Form the rays
        ray_d = ray_o2 - ray_o
        ray = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        ray.prop_to(depth)
        return ray

    # ====================================================================================
    # Lens 2D layout
    # ====================================================================================
    def _surface_material_label(self, surface):
        """Return a display label for the material after a surface."""
        label = getattr(surface, "material_display_name", None)
        if not label:
            label = surface.mat2.get_name()
        if label is None:
            return "air"
        return str(label).upper()

    def _material_color(self, label):
        """Assign stable, readable colors to glass materials."""
        material_palette = {
            "CAF2": "#8DD3C7",
            "N-BK7": "#80B1D3",
            "N-SF11": "#FB8072",
            "C79-80": "#B3DE69",
            "AIR": "#FFFFFF",
        }
        if label in material_palette:
            return material_palette[label]

        fallback_palette = [
            "#FDB462",
            "#BEBADA",
            "#FCCDE5",
            "#BC80BD",
            "#CCEBC5",
            "#FFED6F",
            "#A6CEE3",
            "#FBB4AE",
        ]
        return fallback_palette[sum(ord(ch) for ch in label) % len(fallback_palette)]

    def _draw_lens_fill_2d(self, ax, s_prev, s, fill_color):
        """Fill the 2D cross-section between two adjacent refracting surfaces."""
        r_prev = float(s_prev.r)
        r = float(s.r)

        x_prev = torch.linspace(-r_prev, r_prev, 128, device=s_prev.device)
        z_prev = s_prev.surface_with_offset(
            x_prev, torch.zeros(len(x_prev), device=s_prev.device)
        )

        x_curr = torch.linspace(r, -r, 128, device=s.device)
        z_curr = s.surface_with_offset(
            x_curr, torch.zeros(len(x_curr), device=s.device)
        )

        z_poly = np.concatenate(
            [z_prev.cpu().detach().numpy(), z_curr.cpu().detach().numpy()]
        )
        x_poly = np.concatenate(
            [x_prev.cpu().detach().numpy(), x_curr.cpu().detach().numpy()]
        )

        ax.fill(
            z_poly,
            x_poly,
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.38,
            zorder=0,
        )

    def _draw_scale_bar_2d(self, ax):
        """Draw a horizontal millimeter scale bar in the 2D lens layout."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_span = float(x_max - x_min)
        y_span = float(y_max - y_min)
        if x_span <= 0 or y_span <= 0:
            return None

        raw_len = x_span * 0.15
        if raw_len <= 0:
            return None

        base = 10 ** np.floor(np.log10(raw_len))
        scale_len = base
        for factor in (5, 2, 1):
            candidate = factor * base
            if candidate <= raw_len:
                scale_len = candidate
                break

        x0 = x_min + 0.06 * x_span
        y0 = y_min + 0.09 * y_span
        tick = 0.025 * y_span
        label = f"{scale_len:g} mm"

        ax.plot(
            [x0, x0 + scale_len],
            [y0, y0],
            color="black",
            linewidth=1.4,
            solid_capstyle="butt",
            zorder=20,
        )
        ax.plot(
            [x0, x0],
            [y0 - tick / 2, y0 + tick / 2],
            color="black",
            linewidth=1.2,
            zorder=20,
        )
        ax.plot(
            [x0 + scale_len, x0 + scale_len],
            [y0 - tick / 2, y0 + tick / 2],
            color="black",
            linewidth=1.2,
            zorder=20,
        )
        ax.text(
            x0 + scale_len / 2,
            y0 + 0.035 * y_span,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=20,
        )
        return x0, y0, scale_len

    def _draw_bottom_legends_2d(self, ax, scale_bar, field_items, material_handles):
        """Draw compact horizontal legends to the right of the scale bar."""
        if scale_bar is None:
            return

        x_min, x_max = ax.get_xlim()
        x_span = float(x_max - x_min)
        if x_span <= 0:
            return

        scale_x0, scale_y0, scale_len = scale_bar
        start_x = (scale_x0 - x_min + scale_len) / x_span + 0.045
        start_x = min(max(start_x, 0.20), 0.36)
        field_y = 0.155
        material_y = 0.045
        font_size = 8.0
        line_len = 0.040
        field_label_gap = 0.110
        field_step = 0.120
        box_w = 0.024
        box_h = 0.036
        material_label_gap = 0.150
        material_step = 0.122

        field_width = field_label_gap + len(field_items) * (
            line_len + 0.010 + field_step
        )
        material_width = 0.0
        if material_handles:
            material_width = material_label_gap + len(material_handles) * (
                box_w + 0.010 + material_step
            )
        row_width = max(field_width, material_width)
        center_x = start_x + row_width / 2
        center_x = min(center_x, 0.98 - row_width / 2)
        center_x = max(center_x, start_x + row_width / 2)
        field_start_x = center_x - field_width / 2
        material_start_x = center_x - material_width / 2

        def draw_text(x, y, text, weight="normal"):
            ax.text(
                x,
                y,
                text,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=font_size,
                fontweight=weight,
                color="black",
                zorder=25,
                clip_on=False,
            )

        x = field_start_x
        draw_text(x, field_y, "Fields:", weight="bold")
        x += field_label_gap
        for label, color in field_items:
            ax.plot(
                [x, x + line_len],
                [field_y, field_y],
                color=color,
                linewidth=2.0,
                solid_capstyle="butt",
                transform=ax.transAxes,
                zorder=25,
                clip_on=False,
            )
            x += line_len + 0.010
            draw_text(x, field_y, label)
            x += field_step

        if not material_handles:
            return

        x = material_start_x
        draw_text(x, material_y, "Materials:", weight="bold")
        x += material_label_gap
        for handle in material_handles:
            label = handle.get_label()
            patch = mpatches.Rectangle(
                (x, material_y - box_h / 2),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor=handle.get_facecolor(),
                edgecolor="black",
                linewidth=0.45,
                alpha=0.55,
                zorder=25,
                clip_on=False,
            )
            ax.add_patch(patch)
            x += box_w + 0.010
            draw_text(x, material_y, label)
            x += material_step

    def draw_layout(
        self,
        filename,
        depth=float("inf"),
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
        show=False,
    ):
        """Plot 2D lens layout with ray tracing.

        Args:
            filename: Output filename
            depth: Depth for ray tracing
            entrance_pupil: Whether to use entrance pupil
            zmx_format: Whether to use ZMX format
            multi_plot: Whether to create multiple plots
            lens_title: Title for the lens plot
            show: Whether to show the plot
        """
        num_rays = 3
        num_views = 3

        # Lens title
        if lens_title is None:
            eff_foclen = int(self.foclen)
            eq_foclen = int(self.eqfl)
            fov_deg = round(2 * self.rfov * 180 / torch.pi, 1)
            sensor_r = round(self.r_sensor, 1)
            sensor_w, sensor_h = self.sensor_size
            sensor_w = round(sensor_w, 1)
            sensor_h = round(sensor_h, 1)

            if self.aper_idx is not None:
                _, pupil_r = self.calc_entrance_pupil()
                fnum = round(eff_foclen / pupil_r / 2, 1)
                lens_title = f"FocLen{eff_foclen}mm - F/{fnum} - FoV{fov_deg}(Equivalent {eq_foclen}mm) - Sensor Diagonal {2 * sensor_r}mm"
            else:
                lens_title = f"FocLen{eff_foclen}mm - FoV{fov_deg}(Equivalent {eq_foclen}mm) - Sensor Diagonal {2 * sensor_r}mm"

        # Draw lens layout
        colors_list = ["#CC0000", "#006600", "#0066CC"]
        rfov_deg = float(np.rad2deg(self.rfov))
        fov_ls = np.linspace(0, rfov_deg * 0.99, num=num_views)
        
        if not multi_plot:
            ax, fig = self.draw_lens_2d(zmx_format=zmx_format)
            fig.suptitle(lens_title, fontsize=10)
            for i, fov in enumerate(fov_ls):
                # Sample rays, shape (num_rays, 3)
                if depth == float("inf"):
                    ray = self.sample_parallel_2D(
                        fov=fov,
                        wvln=WAVE_RGB[2 - i],
                        num_rays=num_rays,
                        depth=-1.0,
                        plane="sagittal",
                    )
                else:
                    ray = self.sample_point_source_2D(
                        fov=fov,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=WAVE_RGB[2 - i],
                    )
                    ray.prop_to(-1.0)

                # Trace rays to sensor and plot ray paths
                _, ray_o_record = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.draw_ray_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )

            # Tighten the vertical extent so the layout is not dominated by
            # the whitespace that equal-aspect autoscaling introduces.
            ax.set_aspect("equal", adjustable="box")
            ylim = float(self.r_sensor) * 1.05
            ax.set_ylim(-ylim * 1.65, ylim)
            scale_bar = self._draw_scale_bar_2d(ax)

            field_names = ["center", "mid", "edge"]
            field_items = [
                (field_names[i] if i < len(field_names) else f"field {i}", colors_list[i])
                for i in range(num_views)
            ]
            material_handles = getattr(self, "_layout_material_legend_handles", [])
            self._draw_bottom_legends_2d(ax, scale_bar, field_items, material_handles)

            ax.axis("off")

        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(lens_title, fontsize=10)
            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.draw_lens_2d(ax=ax, fig=fig, zmx_format=zmx_format)
                for fov in fov_ls:
                    # Sample rays, shape (num_rays, 3)
                    if depth == float("inf"):
                        ray = self.sample_parallel_2D(
                            fov=fov,
                            num_rays=num_rays,
                            wvln=wvln,
                            plane="sagittal",
                        )
                    else:
                        ray = self.sample_point_source_2D(
                            fov=fov,
                            depth=depth,
                            num_rays=num_rays,
                            wvln=wvln,
                        )

                    # Trace rays to sensor and plot ray paths
                    ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.draw_ray_2d(
                        ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                    )
                    ax.axis("off")

        if show:
            fig.show()
        else:
            fig.savefig(filename, format="png", dpi=300)
            plt.close()

    def draw_lens_2d(
        self,
        ax=None,
        fig=None,
        color="k",
        linestyle="-",
        zmx_format=False,
        fix_bound=False,
    ):
        """Draw lens layout in a 2D plot."""
        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            # fig, ax = plt.subplots(figsize=(6, 6))
            fig, ax = plt.subplots()

        material_handles_by_label = {}

        # Fill glass bodies by material before drawing outlines.
        for i in range(len(self.surfaces) - 1):
            if self.surfaces[i].mat2.n > 1.1:
                material_label = self._surface_material_label(self.surfaces[i])
                fill_color = self._material_color(material_label)
                self._draw_lens_fill_2d(
                    ax,
                    self.surfaces[i],
                    self.surfaces[i + 1],
                    fill_color,
                )
                material_handles_by_label.setdefault(
                    material_label,
                    mpatches.Patch(
                        facecolor=fill_color,
                        edgecolor="black",
                        linewidth=0.4,
                        alpha=0.55,
                        label=material_label,
                    ),
                )

        self._layout_material_legend_handles = list(
            material_handles_by_label.values()
        )

        # Draw lens surfaces
        for i, s in enumerate(self.surfaces):
            s.draw_widget(ax)

        # Connect two surfaces
        for i in range(len(self.surfaces) - 1):
            if self.surfaces[i].mat2.n > 1.1:
                s_prev = self.surfaces[i]
                s = self.surfaces[i + 1]

                r_prev = float(s_prev.r)
                r = float(s.r)
                sag_prev = s_prev.surface_with_offset(r_prev, 0.0).item()
                sag = s.surface_with_offset(r, 0.0).item()

                if zmx_format:
                    if r > r_prev:
                        z = np.array([sag_prev, sag_prev, sag])
                        x = np.array([r_prev, r, r])
                    else:
                        z = np.array([sag_prev, sag, sag])
                        x = np.array([r_prev, r, r])
                else:
                    z = np.array([sag_prev, sag])
                    x = np.array([r_prev, r])

                ax.plot(z, -x, color, linewidth=0.75)
                ax.plot(z, x, color, linewidth=0.75)
                s_prev = s

        # Draw sensor
        ax.plot(
            [self.d_sensor.item(), self.d_sensor.item()],
            [-self.r_sensor, self.r_sensor],
            color,
        )

        # Set figure size
        if fix_bound:
            ax.set_aspect("equal")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect("equal", adjustable="datalim", anchor="C")
            ax.minorticks_on()
            ax.set_xlim(-0.5, 7.5)
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig

    def draw_ray_2d(self, ray_o_record, ax, fig, color="b"):
        """Plot ray paths.

        Args:
            ray_o_record (list): list of intersection points.
            ax (matplotlib.axes.Axes): matplotlib axes.
            fig (matplotlib.figure.Figure): matplotlib figure.
        """
        # shape (num_view, num_rays, num_path, 2)
        ray_o_record = torch.stack(ray_o_record, dim=-2).cpu().numpy()
        if ray_o_record.ndim == 3:
            ray_o_record = ray_o_record[None, ...]

        for idx_view in range(ray_o_record.shape[0]):
            for idx_ray in range(ray_o_record.shape[1]):
                ax.plot(
                    ray_o_record[idx_view, idx_ray, :, 2],
                    ray_o_record[idx_view, idx_ray, :, 0],
                    color,
                    linewidth=0.8,
                )

                # ax.scatter(
                #     ray_o_record[idx_view, idx_ray, :, 2],
                #     ray_o_record[idx_view, idx_ray, :, 0],
                #     "b",
                #     marker="x",
                # )

        return ax, fig

    # ====================================================================================
    # Lens 3D layout
    # ====================================================================================
    def draw_layout_3d(self, filename=None, view_angle=30, show=False):
        """Draw 3D layout of the lens system.

        Args:
            filename (str, optional): Path to save the figure. Defaults to None.
            view_angle (int): Viewing angle for the 3D plot
            show (bool): Whether to display the figure

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        raise Exception(
            "This function is deprecated. Please use the draw_lens_3d function in the view_3d module instead."
        )
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Enable depth sorting for proper occlusion
        ax.set_proj_type(
            "persp"
        )  # Use perspective projection for better depth perception

        # Draw each surface
        for i, surf in enumerate(self.surfaces):
            surf.draw_widget3D(ax)

            # Connect current surface with previous surface if material is not air
            if i > 0 and self.surfaces[i - 1].mat2.get_name() != "air":
                # Get edge points of current and previous surfaces
                theta = np.linspace(0, 2 * np.pi, 256)

                # Current surface edge
                curr_edge_x = surf.r * np.cos(theta)
                curr_edge_y = surf.r * np.sin(theta)
                curr_edge_z = np.array(
                    [
                        surf.surface_with_offset(
                            torch.tensor(curr_edge_x[j], device=surf.device),
                            torch.tensor(curr_edge_y[j], device=surf.device),
                        ).item()
                        for j in range(len(theta))
                    ]
                )

                # Previous surface edge
                prev_surf = self.surfaces[i - 1]
                prev_edge_x = prev_surf.r * np.cos(theta)
                prev_edge_y = prev_surf.r * np.sin(theta)
                prev_edge_z = np.array(
                    [
                        prev_surf.surface_with_offset(
                            torch.tensor(prev_edge_x[j], device=prev_surf.device),
                            torch.tensor(prev_edge_y[j], device=prev_surf.device),
                        ).item()
                        for j in range(len(theta))
                    ]
                )

                # Create a cylindrical surface connecting the two edges
                theta_mesh, t_mesh = np.meshgrid(theta, np.array([0, 1]))

                # Interpolate between previous and current surface edges
                x_mesh = (
                    prev_edge_x[None, :] * (1 - t_mesh) + curr_edge_x[None, :] * t_mesh
                )
                y_mesh = (
                    prev_edge_y[None, :] * (1 - t_mesh) + curr_edge_y[None, :] * t_mesh
                )
                z_mesh = (
                    prev_edge_z[None, :] * (1 - t_mesh) + curr_edge_z[None, :] * t_mesh
                )

                # Plot the connecting surface with sort_zpos for proper occlusion
                surf = ax.plot_surface(
                    z_mesh,
                    x_mesh,
                    y_mesh,
                    color="lightblue",
                    alpha=0.3,
                    edgecolor="lightblue",
                    linewidth=0.5,
                    antialiased=True,
                )
                # Set the zorder based on the mean z position for better occlusion
                surf._sort_zpos = np.mean(z_mesh)

        # Draw sensor as a rectangle
        if hasattr(self, "sensor_size") and hasattr(self, "d_sensor"):
            # Get sensor dimensions
            sensor_width = self.sensor_size[0]
            sensor_height = self.sensor_size[1]
            sensor_z = self.d_sensor.item()

            # Create sensor vertices
            half_width = sensor_width / 2
            half_height = sensor_height / 2

            # Define the corners of the rectangle
            x = np.array(
                [-half_width, half_width, half_width, -half_width, -half_width]
            )
            y = np.array(
                [-half_height, -half_height, half_height, half_height, -half_height]
            )
            z = np.full_like(x, sensor_z)

            # Plot the sensor rectangle
            ax.plot(z, x, y, color="black", linewidth=1.5)

            # Add a semi-transparent surface for the sensor
            sensor_x, sensor_y = np.meshgrid(
                np.linspace(-half_width, half_width, 2),
                np.linspace(-half_height, half_height, 2),
            )
            sensor_z = np.full_like(sensor_x, sensor_z)
            sensor_surf = ax.plot_surface(
                sensor_z,
                sensor_x,
                sensor_y,
                color="gray",
                alpha=0.3,
                edgecolor="black",
                linewidth=0.5,
            )
            # Set the zorder for the sensor
            sensor_surf._sort_zpos = sensor_z.mean()

        # Set axis properties
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        ax.view_init(elev=20, azim=-view_angle - 90)

        # Make all axes have the same scale (unit step size)
        ax.set_box_aspect([1, 1, 1])
        ax.set_aspect("equal")

        # Enable depth sorting for proper occlusion
        from matplotlib.collections import PathCollection

        for c in ax.collections:
            if isinstance(c, PathCollection):
                c.set_sort_zpos(c.get_offsets()[:, 2].mean())

        plt.tight_layout()

        if filename:
            fig.savefig(f"{filename}.png", format="png", dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    # ====================================================================================
    # Lens 3D barrier generation
    # ====================================================================================
    def create_barrier(
        self, filename, barrier_thickness=1.0, ring_height=0.5, ring_size=1.0
    ):
        """Create a 3D barrier for the lens system.

        Args:
            filename: Path to save the figure
            barrier_thickness: Thickness of the barrier
            ring_height: Height of the annular ring
            ring_size: Size of the annular ring
        """
        barriers = []
        rings = []

        # Create barriers
        barrier_z = 0.0
        barrier_r = 0.0
        barrier_length = 0.0
        for i in range(len(self.surfaces)):
            barrier_r = max(self.surfaces[i].r, barrier_r)

            if self.surfaces[i].mat2.get_name() != "air":
                # Update the barrier radius
                # barrier_r = max(geolens.surfaces[i].r, barrier_r)
                pass
            else:
                # Extend the barrier till middle of the air space to the next surface
                max_curr_surf_d = self.surfaces[i].d.item() + max(
                    self.surfaces[i].surface_sag(0.0, self.surfaces[i].r), 0.0
                )
                if i < len(self.surfaces) - 1:
                    min_next_surf_d = self.surfaces[i + 1].d.item() + min(
                        self.surfaces[i + 1].surface_sag(0.0, self.surfaces[i + 1].r),
                        0.0,
                    )
                    extra_space = (min_next_surf_d - max_curr_surf_d) / 2
                else:
                    min_next_surf_d = self.d_sensor.item()
                    extra_space = min_next_surf_d - max_curr_surf_d

                barrier_length = max_curr_surf_d + extra_space - barrier_z

                # Create a barrier
                barrier = {
                    "pos_z": barrier_z,
                    "pos_r": barrier_r,
                    "length": barrier_length,
                    "thickness": barrier_thickness,
                }
                barriers.append(barrier)

                # Reset the barrier parameters
                barrier_z = barrier_length + barrier_z
                barrier_r = 0.0
                barrier_length = 0.0

        # # Create rings
        # for i in range(len(geolens.surfaces)):
        #     if geolens.surfaces[i].mat2.get_name() != "air":
        #         ring = {
        #             "pos_z": geolens.surfaces[i].d.item(),

        # Plot lens layout
        ax, fig = self.draw_layout()

        # Plot barrier
        barrier_z_ls = []
        barrier_r_ls = []
        for b in barriers:
            barrier_z_ls.append(b["pos_z"])
            barrier_z_ls.append(b["pos_z"] + b["length"])
            barrier_r_ls.append(b["pos_r"])
            barrier_r_ls.append(b["pos_r"])
        ax.plot(barrier_z_ls, barrier_r_ls, "green", linewidth=1.0)
        ax.plot(barrier_z_ls, [-i for i in barrier_r_ls], "green", linewidth=1.0)

        # Plot rings

        fig.savefig(filename, format="png", dpi=300)
        plt.close()

        pass
