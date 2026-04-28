use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use vello::kurbo::{Affine, Rect, RoundedRect, Stroke, Vec2};
use vello::peniko::{Color, Fill};
use vello::util::{RenderContext, RenderSurface};
use vello::wgpu;
use vello::{AaConfig, Renderer, RendererOptions, Scene};
use vello_svg::usvg;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::Window;

#[cfg(feature = "dhat")]
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat")]
fn mem_log(label: &str) {
    let bytes = unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return;
        }
        // macOS reports ru_maxrss in bytes; Linux reports kilobytes.
        if cfg!(target_os = "macos") {
            ru.ru_maxrss as f64
        } else {
            ru.ru_maxrss as f64 * 1024.0
        }
    };
    eprintln!(
        "mem [{label}]: rss_peak={:.1} MiB",
        bytes / (1024.0 * 1024.0)
    );
}

#[cfg(not(feature = "dhat"))]
fn mem_log(_label: &str) {}

const ZOOM_PER_LINE: f64 = 1.15;
const ZOOM_PER_PIXEL: f64 = 0.0015;
const PAN_STEP: f64 = 40.0;
const PAN_STEP_LARGE: f64 = 160.0;
const MIN_SCALE: f64 = 0.05;
const MAX_SCALE: f64 = 1000.0;

#[derive(Debug, Clone, Copy)]
enum AppEvent {
    FileChanged,
}

enum RenderState {
    Active {
        surface: Box<RenderSurface<'static>>,
        window: Arc<Window>,
    },
    Suspended(Option<Arc<Window>>),
}

struct App {
    context: RenderContext,
    renderers: Vec<Option<Renderer>>,
    state: RenderState,
    scene: Scene,
    fragment: Scene,
    svg_path: PathBuf,
    svg_size: Vec2,
    transform: Affine,
    cursor: PhysicalPosition<f64>,
    middle_pan_anchor: Option<PhysicalPosition<f64>>,
    modifiers: ModifiersState,
    /// Lock the trackpad scroll session to its initial mode (Some(true) =
    /// zooming, Some(false) = panning) so releasing/pressing Cmd mid-gesture
    /// doesn't flip behavior. None when no gesture is active.
    scroll_lock_zoom: Option<bool>,
    proxy: EventLoopProxy<AppEvent>,
    fontdb: Arc<usvg::fontdb::Database>,
    text_index: Vec<TextEntry>,
    search_active: bool,
    search_query: String,
    matches: Vec<usize>,
    current_match: Option<usize>,
    /// Set when Cmd+F is pressed so the next egui frame focuses the input.
    want_focus_search: bool,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
    _watcher: Option<notify::RecommendedWatcher>,
}

#[derive(Debug, Clone)]
struct TextEntry {
    text: String,
    bounds: Rect,
}

/// Replace egui's bundled font with a system sans-serif (matches the OS UI).
fn install_system_font(ctx: &egui::Context, db: &usvg::fontdb::Database) {
    use usvg::fontdb::{Family, Query};
    let candidates: [&[Family]; 6] = [
        &[Family::Name("SF Pro Text"), Family::SansSerif],
        &[Family::Name("Helvetica Neue"), Family::SansSerif],
        &[Family::Name("Helvetica"), Family::SansSerif],
        &[Family::Name("Arial"), Family::SansSerif],
        &[Family::Name("DejaVu Sans"), Family::SansSerif],
        &[Family::SansSerif],
    ];
    for fams in candidates {
        let q = Query {
            families: fams,
            ..Query::default()
        };
        let Some(id) = db.query(&q) else { continue };
        let Some(bytes) = db.with_face_data(id, |data, _| data.to_vec()) else {
            continue;
        };
        let mut defs = egui::FontDefinitions::default();
        let name = "system-ui".to_string();
        defs.font_data
            .insert(name.clone(), Arc::new(egui::FontData::from_owned(bytes)));
        defs.families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, name.clone());
        defs.families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, name);
        ctx.set_fonts(defs);
        return;
    }
    log::warn!("no system sans-serif font found; egui will use its bundled font");
}

impl App {
    fn new(svg_path: PathBuf, proxy: EventLoopProxy<AppEvent>) -> Result<Self> {
        mem_log("App::new entry");
        // Loading system fonts is the slowest startup step (~hundreds of ms
        // on macOS); do it once and share the Arc across reloads.
        let t0 = Instant::now();
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        log::info!(
            "loaded {} system font faces in {:?}",
            db.len(),
            t0.elapsed()
        );
        mem_log("after load_system_fonts");

        let egui_ctx = egui::Context::default();
        install_system_font(&egui_ctx, &db);
        mem_log("after egui ctx + system font");

        let mut app = Self {
            context: RenderContext::new(),
            renderers: vec![],
            state: RenderState::Suspended(None),
            scene: Scene::new(),
            fragment: Scene::new(),
            svg_path,
            svg_size: Vec2::new(1.0, 1.0),
            transform: Affine::IDENTITY,
            cursor: PhysicalPosition::new(0.0, 0.0),
            middle_pan_anchor: None,
            modifiers: ModifiersState::empty(),
            scroll_lock_zoom: None,
            proxy,
            fontdb: Arc::new(db),
            text_index: Vec::new(),
            search_active: false,
            search_query: String::new(),
            matches: Vec::new(),
            current_match: None,
            want_focus_search: false,
            egui_ctx,
            egui_state: None,
            egui_renderer: None,
            _watcher: None,
        };
        app.load_svg().context("loading initial SVG")?;
        mem_log("after initial load_svg");
        app.start_watcher();
        Ok(app)
    }

    fn start_watcher(&mut self) {
        let proxy = self.proxy.clone();
        let watch_file = self.svg_path.clone();
        let watch_name = self.svg_path.file_name().map(|n| n.to_os_string());
        let watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let Ok(ev) = res else { return };
            if !matches!(
                ev.kind,
                notify::EventKind::Modify(_) | notify::EventKind::Create(_)
            ) {
                return;
            }
            // Watching the parent dir surfaces events for unrelated siblings
            // (editor swap files, etc.); only react to our SVG.
            let touches_target = ev.paths.iter().any(|p| {
                p == &watch_file
                    || p.canonicalize().ok().as_deref() == Some(watch_file.as_path())
                    || (watch_name.is_some() && p.file_name() == watch_name.as_deref())
            });
            if touches_target {
                let _ = proxy.send_event(AppEvent::FileChanged);
            }
        });
        if let Ok(mut w) = watcher {
            // Watch the parent dir to survive editor "atomic save" rename patterns.
            let target = self
                .svg_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("."));
            match w.watch(&target, RecursiveMode::NonRecursive) {
                Ok(()) => self._watcher = Some(w),
                Err(err) => log::warn!("file watcher failed on {}: {err}", target.display()),
            }
        }
    }

    fn load_svg(&mut self) -> Result<()> {
        let svg = std::fs::read_to_string(&self.svg_path)
            .with_context(|| format!("reading {}", self.svg_path.display()))?;
        let opt = usvg::Options {
            fontdb: self.fontdb.clone(),
            ..Default::default()
        };
        let tree = usvg::Tree::from_str(&svg, &opt).context("parsing SVG")?;
        self.svg_size = Vec2::new(tree.size().width() as f64, tree.size().height() as f64);
        self.fragment = Scene::new();
        vello_svg::append_tree(&mut self.fragment, &tree);

        self.text_index.clear();
        Self::collect_text(tree.root(), &mut self.text_index);
        // Always rebuild matches so a non-empty query reflects the new file
        // even if the search bar is currently closed.
        self.recompute_matches();
        Ok(())
    }

    fn collect_text(group: &usvg::Group, out: &mut Vec<TextEntry>) {
        for node in group.children() {
            match node {
                usvg::Node::Group(g) => Self::collect_text(g, out),
                usvg::Node::Text(t) => {
                    let mut s = String::new();
                    for chunk in t.chunks() {
                        if !s.is_empty() {
                            s.push(' ');
                        }
                        s.push_str(chunk.text());
                    }
                    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
                    if s.is_empty() {
                        continue;
                    }
                    let r = node.abs_bounding_box();
                    let bounds = Rect::new(
                        r.x() as f64,
                        r.y() as f64,
                        (r.x() + r.width()) as f64,
                        (r.y() + r.height()) as f64,
                    );
                    out.push(TextEntry { text: s, bounds });
                }
                _ => {}
            }
        }
    }

    fn open_search(&mut self) {
        self.search_active = true;
        self.want_focus_search = true;
        if !self.search_query.is_empty() {
            self.recompute_matches();
        }
    }

    fn close_search(&mut self) {
        self.search_active = false;
        self.matches.clear();
        self.current_match = None;
    }

    fn recompute_matches(&mut self) {
        self.matches.clear();
        if self.search_query.is_empty() {
            self.current_match = None;
            return;
        }
        let q = self.search_query.to_lowercase();
        for (i, e) in self.text_index.iter().enumerate() {
            if e.text.to_lowercase().contains(&q) {
                self.matches.push(i);
            }
        }
        self.current_match = if self.matches.is_empty() { None } else { Some(0) };
    }

    fn next_match(&mut self) {
        if let Some(i) = self.current_match {
            if !self.matches.is_empty() {
                self.current_match = Some((i + 1) % self.matches.len());
            }
        }
    }

    fn prev_match(&mut self) {
        if let Some(i) = self.current_match {
            if !self.matches.is_empty() {
                self.current_match = Some((i + self.matches.len() - 1) % self.matches.len());
            }
        }
    }

    fn focus_current_match(&mut self, viewport_w: u32, viewport_h: u32) {
        let Some(idx) = self.current_match else { return };
        let Some(entry_idx) = self.matches.get(idx).copied() else { return };
        let bounds = self.text_index[entry_idx].bounds;
        if bounds.width() <= 0.0 || bounds.height() <= 0.0 {
            return;
        }
        let current = self.current_scale();
        // Target ~ 90 physical px tall for the matched text -- comfortable
        // reading size regardless of the SVG's intrinsic units.
        let target_text_px = 90.0_f64;
        let desired = target_text_px / bounds.height();
        // Never zoom past the point where the bbox fills more than 70% of
        // the viewport (avoids absurd zoom-in on huge text blocks).
        let fit_w = (viewport_w as f64 * 0.7) / bounds.width();
        let fit_h = (viewport_h as f64 * 0.7) / bounds.height();
        let fit = fit_w.min(fit_h);
        let scale = current.max(desired).min(fit).max(MIN_SCALE);
        let cx = (bounds.x0 + bounds.x1) * 0.5;
        let cy = (bounds.y0 + bounds.y1) * 0.5;
        let tx = viewport_w as f64 * 0.5 - cx * scale;
        let ty = viewport_h as f64 * 0.5 - cy * scale;
        self.transform = Affine::translate((tx, ty)) * Affine::scale(scale);
    }

    fn current_match_bounds(&self) -> Option<Rect> {
        let idx = self.current_match?;
        let entry_idx = *self.matches.get(idx)?;
        Some(self.text_index[entry_idx].bounds)
    }

    /// Build the egui search bar UI. Returns true if matches need
    /// recomputing or the view needs to refocus on the current match.
    fn build_egui(&mut self, ctx: &egui::Context) -> EguiResult {
        let mut result = EguiResult::default();
        if !self.search_active {
            return result;
        }

        egui::Area::new(egui::Id::new("svgwatch-search"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 40.0))
            .order(egui::Order::Foreground)
            .interactable(true)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .shadow(egui::epaint::Shadow::NONE)
                    .show(ui, |ui| {
                        // All row children share a vertical center.
                        ui.with_layout(
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                let row_h = ui.spacing().interact_size.y;
                                ui.spacing_mut().item_spacing.x = 8.0;

                                ui.add_sized(
                                    egui::vec2(28.0, row_h),
                                    egui::Label::new(
                                        egui::RichText::new("Find")
                                            .strong()
                                            .color(ui.style().visuals.weak_text_color()),
                                    )
                                    .wrap_mode(egui::TextWrapMode::Extend),
                                );

                                let id = egui::Id::new("svgwatch-search-input");
                                let edit =
                                    egui::TextEdit::singleline(&mut self.search_query)
                                        .id(id)
                                        .desired_width(220.0)
                                        .min_size(egui::vec2(220.0, row_h))
                                        .hint_text("type to search…")
                                        .vertical_align(egui::Align::Center);
                                let resp = ui.add(edit);

                                if self.want_focus_search {
                                    resp.request_focus();
                                    self.want_focus_search = false;
                                }
                                if resp.changed() {
                                    result.recompute = true;
                                }

                                // Match counter — fixed-width slot so the
                                // layout never jumps.
                                let counter = if self.search_query.is_empty() {
                                    String::new()
                                } else if self.matches.is_empty() {
                                    "no match".to_string()
                                } else {
                                    format!(
                                        "{} / {}",
                                        self.current_match
                                            .map(|i| i + 1)
                                            .unwrap_or(0),
                                        self.matches.len()
                                    )
                                };
                                let counter_color = if self.matches.is_empty()
                                    && !self.search_query.is_empty()
                                {
                                    egui::Color32::from_rgb(200, 80, 40)
                                } else {
                                    ui.style().visuals.weak_text_color()
                                };
                                ui.add_sized(
                                    egui::vec2(70.0, row_h),
                                    egui::Label::new(
                                        egui::RichText::new(counter).color(counter_color),
                                    )
                                    .wrap_mode(egui::TextWrapMode::Truncate),
                                );

                                // Prev / Next buttons sized to row height.
                                let btn = egui::vec2(row_h, row_h);
                                let prev = ui
                                    .add_enabled_ui(!self.matches.is_empty(), |ui| {
                                        ui.add_sized(btn, egui::Button::new("◀"))
                                    })
                                    .inner
                                    .on_hover_text("Previous (Shift+Enter)");
                                let next = ui
                                    .add_enabled_ui(!self.matches.is_empty(), |ui| {
                                        ui.add_sized(btn, egui::Button::new("▶"))
                                    })
                                    .inner
                                    .on_hover_text("Next (Enter)");
                                if prev.clicked() {
                                    result.prev = true;
                                }
                                if next.clicked() {
                                    result.next = true;
                                }

                                // Keyboard shortcuts within the search field.
                                ui.input(|i| {
                                    if i.key_pressed(egui::Key::Enter) {
                                        if i.modifiers.shift {
                                            result.prev = true;
                                        } else {
                                            result.next = true;
                                        }
                                    }
                                    if i.key_pressed(egui::Key::Escape) {
                                        result.close = true;
                                    }
                                });
                            },
                        );
                    });
            });

        result
    }
}

#[derive(Default, Debug)]
struct EguiResult {
    recompute: bool,
    next: bool,
    prev: bool,
    close: bool,
}

impl App {
    fn fit(&mut self, w: u32, h: u32) {
        if self.svg_size.x <= 0.0 || self.svg_size.y <= 0.0 {
            return;
        }
        let sx = w as f64 / self.svg_size.x;
        let sy = h as f64 / self.svg_size.y;
        let s = sx.min(sy) * 0.95;
        let tx = (w as f64 - self.svg_size.x * s) / 2.0;
        let ty = (h as f64 - self.svg_size.y * s) / 2.0;
        self.transform = Affine::translate((tx, ty)) * Affine::scale(s);
    }

    fn current_scale(&self) -> f64 {
        // Affine columns are (m11, m21, m12, m22, tx, ty); uniform scale assumed.
        let c = self.transform.as_coeffs();
        (c[0] * c[0] + c[1] * c[1]).sqrt()
    }

    fn zoom_at(&mut self, anchor_x: f64, anchor_y: f64, factor: f64) {
        let current = self.current_scale();
        let new_scale = (current * factor).clamp(MIN_SCALE, MAX_SCALE);
        let factor = new_scale / current;
        if (factor - 1.0).abs() < 1e-9 {
            return;
        }
        let p = (anchor_x, anchor_y);
        self.transform = Affine::translate(p)
            * Affine::scale(factor)
            * Affine::translate((-p.0, -p.1))
            * self.transform;
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        self.transform = Affine::translate((dx, dy)) * self.transform;
    }

    fn request_redraw(&self) {
        if let RenderState::Active { window, .. } = &self.state {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached) = &mut self.state else {
            return;
        };
        let window = cached.take().unwrap_or_else(|| {
            #[allow(unused_mut)]
            let mut attr = Window::default_attributes()
                .with_inner_size(LogicalSize::new(1024, 768))
                .with_title("svgwatch");
            #[cfg(target_os = "macos")]
            {
                use winit::platform::macos::WindowAttributesExtMacOS;
                attr = attr
                    .with_titlebar_transparent(true)
                    .with_title_hidden(true)
                    .with_fullsize_content_view(true);
            }
            Arc::new(event_loop.create_window(attr).unwrap())
        });

        let size = window.inner_size();
        let surface = pollster::block_on(self.context.create_surface(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            wgpu::PresentMode::AutoVsync,
        ))
        .expect("create surface");

        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id].get_or_insert_with(|| {
            // Only `AaConfig::Area` is used at render time; compiling MSAA8 and
            // MSAA16 pipelines wastes ~hundreds of MB of GPU memory.
            Renderer::new(
                &self.context.devices[surface.dev_id].device,
                RendererOptions {
                    use_cpu: false,
                    antialiasing_support: vello::AaSupport::area_only(),
                    num_init_threads: std::num::NonZeroUsize::new(1),
                    pipeline_cache: None,
                },
            )
            .expect("create renderer")
        });
        mem_log("after Renderer::new");

        self.fit(size.width, size.height);

        // egui state and renderer share the wgpu device with vello.
        let dev = &self.context.devices[surface.dev_id].device;
        let renderer = egui_wgpu::Renderer::new(
            dev,
            surface.format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: true,
                predictable_texture_filtering: false,
            },
        );
        self.egui_renderer = Some(renderer);

        let dpr = window.scale_factor() as f32;
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(dpr),
            None,
            Some(8192),
        ));

        self.state = RenderState::Active {
            surface: Box::new(surface),
            window,
        };
        mem_log("after resumed (renderer + egui ready)");
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.state {
            self.state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::FileChanged => {
                if let Err(err) = self.load_svg() {
                    log::warn!("reload failed: {err:#}");
                } else {
                    self.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Borrow only briefly: clone the Arc<Window> so the rest of the
        // handler can mutate `self` freely.
        let window = match &self.state {
            RenderState::Active { window, .. } if window.id() == window_id => {
                window.clone()
            }
            _ => return,
        };

        // Route the event through egui first; if a focused widget consumed
        // it (e.g. the search field is taking keystrokes), don't run our
        // own handlers below.
        let mut consumed_by_egui = false;
        if let Some(state) = self.egui_state.as_mut() {
            let resp = state.on_window_event(&window, &event);
            if resp.repaint {
                window.request_redraw();
            }
            consumed_by_egui = resp.consumed;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let RenderState::Active { surface, .. } = &mut self.state {
                        self.context
                            .resize_surface(surface, size.width, size.height);
                    }
                    window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = position;
                if let Some(prev) = self.middle_pan_anchor {
                    let dx = position.x - prev.x;
                    let dy = position.y - prev.y;
                    self.middle_pan_anchor = Some(position);
                    self.pan(dx, dy);
                    window.request_redraw();
                }
            }

            WindowEvent::MouseInput { state, button, .. } => match (state, button) {
                (ElementState::Pressed, MouseButton::Middle) => {
                    self.middle_pan_anchor = Some(self.cursor);
                }
                (ElementState::Released, MouseButton::Middle) => {
                    self.middle_pan_anchor = None;
                }
                _ => {}
            },

            WindowEvent::MouseWheel { delta, phase, .. } => {
                let cmd = self.modifiers.control_key() || self.modifiers.super_key();
                match delta {
                    MouseScrollDelta::PixelDelta(d) => {
                        // Lock the gesture mode at the start of a trackpad
                        // scroll session: releasing/pressing Cmd mid-gesture
                        // must NOT flip pan<->zoom.
                        let zoom_mode = match phase {
                            TouchPhase::Started => {
                                self.scroll_lock_zoom = Some(cmd);
                                cmd
                            }
                            TouchPhase::Moved => self.scroll_lock_zoom.unwrap_or(cmd),
                            TouchPhase::Ended | TouchPhase::Cancelled => {
                                let z = self.scroll_lock_zoom.unwrap_or(cmd);
                                self.scroll_lock_zoom = None;
                                z
                            }
                        };
                        if zoom_mode {
                            let factor = (d.y * ZOOM_PER_PIXEL).exp();
                            self.zoom_at(self.cursor.x, self.cursor.y, factor);
                        } else {
                            self.pan(d.x, d.y);
                        }
                    }
                    MouseScrollDelta::LineDelta(_, y) => {
                        // Mouse wheel: zoom around cursor.
                        let factor = ZOOM_PER_LINE.powf(y as f64);
                        self.zoom_at(self.cursor.x, self.cursor.y, factor);
                    }
                }
                window.request_redraw();
            }

            WindowEvent::PinchGesture { delta, phase, .. } => {
                if matches!(phase, TouchPhase::Started | TouchPhase::Moved) {
                    let factor = 1.0 + delta;
                    self.zoom_at(self.cursor.x, self.cursor.y, factor);
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                let cmd = self.modifiers.control_key() || self.modifiers.super_key();
                let shift = self.modifiers.shift_key();

                // Ctrl+C / Ctrl+D quit the app (alongside the native window
                // close button). Ctrl, not Cmd, so they match terminal habits.
                if self.modifiers.control_key() {
                    if let Key::Character(c) = &event.logical_key {
                        if c.eq_ignore_ascii_case("c") || c.eq_ignore_ascii_case("d") {
                            event_loop.exit();
                            return;
                        }
                    }
                }

                // Cmd+F always opens search, even if egui has the focused
                // text field (re-focusing is a no-op then).
                if cmd && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("f"))
                {
                    self.open_search();
                    window.request_redraw();
                    return;
                }

                // While search owns input, let egui handle everything else
                // (typing, Enter, Esc, etc.).
                if consumed_by_egui {
                    return;
                }

                let step = if shift { PAN_STEP_LARGE } else { PAN_STEP };
                let mut dirty = true;
                match &event.logical_key {
                    Key::Named(NamedKey::ArrowLeft) => self.pan(step, 0.0),
                    Key::Named(NamedKey::ArrowRight) => self.pan(-step, 0.0),
                    Key::Named(NamedKey::ArrowUp) => self.pan(0.0, step),
                    Key::Named(NamedKey::ArrowDown) => self.pan(0.0, -step),
                    Key::Character(c) => match c.as_str() {
                        "h" | "H" => self.pan(step, 0.0),
                        "l" | "L" => self.pan(-step, 0.0),
                        "k" | "K" => self.pan(0.0, step),
                        "j" | "J" => self.pan(0.0, -step),
                        "f" | "F" => {
                            let s = window.inner_size();
                            self.fit(s.width, s.height);
                        }
                        "0" => {
                            let s = window.inner_size();
                            self.fit(s.width, s.height);
                        }
                        "=" | "+" => self.zoom_at(self.cursor.x, self.cursor.y, 1.25),
                        "-" | "_" => self.zoom_at(self.cursor.x, self.cursor.y, 1.0 / 1.25),
                        _ => dirty = false,
                    },
                    _ => dirty = false,
                }
                if dirty {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                // 1. Build the vello scene: SVG + match highlight overlay.
                self.scene.reset();
                self.scene.append(&self.fragment, Some(self.transform));

                if self.search_active {
                    if let Some(b) = self.current_match_bounds() {
                        let pad = 2.0;
                        let rect = RoundedRect::new(
                            b.x0 - pad,
                            b.y0 - pad,
                            b.x1 + pad,
                            b.y1 + pad,
                            3.0,
                        );
                        self.scene.fill(
                            Fill::NonZero,
                            self.transform,
                            Color::from_rgba8(255, 200, 0, 70),
                            None,
                            &rect,
                        );
                        self.scene.stroke(
                            &Stroke::new(2.0 / self.current_scale().max(1e-6)),
                            self.transform,
                            Color::from_rgba8(255, 140, 0, 230),
                            None,
                            &rect,
                        );
                    }
                }

                // 2. Run egui to build the search bar UI for this frame.
                let raw_input = self
                    .egui_state
                    .as_mut()
                    .expect("egui state")
                    .take_egui_input(&window);
                let ctx = self.egui_ctx.clone();
                let mut egui_result = EguiResult::default();
                let full_output = ctx.run(raw_input, |ctx| {
                    egui_result = self.build_egui(ctx);
                });

                if egui_result.recompute {
                    self.recompute_matches();
                    let s = window.inner_size();
                    self.focus_current_match(s.width, s.height);
                }
                if egui_result.next {
                    self.next_match();
                    let s = window.inner_size();
                    self.focus_current_match(s.width, s.height);
                }
                if egui_result.prev {
                    self.prev_match();
                    let s = window.inner_size();
                    self.focus_current_match(s.width, s.height);
                }
                if egui_result.close {
                    self.close_search();
                }

                self.egui_state
                    .as_mut()
                    .unwrap()
                    .handle_platform_output(&window, full_output.platform_output);

                let pixels_per_point = self.egui_ctx.pixels_per_point();
                let paint_jobs = self
                    .egui_ctx
                    .tessellate(full_output.shapes, pixels_per_point);

                // 3. Render vello → surface, then egui → surface (on top).
                let (width, height) = match &self.state {
                    RenderState::Active { surface, .. } => {
                        (surface.config.width, surface.config.height)
                    }
                    _ => return,
                };

                let RenderState::Active { surface, .. } = &mut self.state else {
                    return;
                };
                let dev_id = surface.dev_id;
                let device_handle = &self.context.devices[dev_id];

                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("get current texture");
                let surface_view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                self.renderers[dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_texture(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &surface.target_view,
                        &vello::RenderParams {
                            base_color: Color::from_rgba8(255, 255, 255, 255),
                            width,
                            height,
                            antialiasing_method: AaConfig::Area,
                        },
                    )
                    .expect("render");

                let mut encoder = device_handle.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("Frame"),
                    },
                );
                surface.blitter.copy(
                    &device_handle.device,
                    &mut encoder,
                    &surface.target_view,
                    &surface_view,
                );

                // egui pass.
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [width, height],
                    pixels_per_point,
                };
                let renderer = self.egui_renderer.as_mut().expect("egui renderer");
                for (id, image_delta) in &full_output.textures_delta.set {
                    renderer.update_texture(
                        &device_handle.device,
                        &device_handle.queue,
                        *id,
                        image_delta,
                    );
                }
                let user_cbufs = renderer.update_buffers(
                    &device_handle.device,
                    &device_handle.queue,
                    &mut encoder,
                    &paint_jobs,
                    &screen_descriptor,
                );
                {
                    let mut rpass = encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &surface_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        })
                        .forget_lifetime();
                    renderer.render(&mut rpass, &paint_jobs, &screen_descriptor);
                }

                let mut all_cbufs = user_cbufs;
                all_cbufs.push(encoder.finish());
                device_handle.queue.submit(all_cbufs);
                for id in &full_output.textures_delta.free {
                    renderer.free_texture(id);
                }

                surface_texture.present();
                let _ = device_handle.device.poll(wgpu::PollType::Poll);

                #[cfg(feature = "dhat")]
                {
                    static FRAME: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let n = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if matches!(n, 0 | 5 | 30 | 120) {
                        mem_log(&format!("after frame {}", n + 1));
                    }
                }

                if full_output.viewport_output.values().any(|v| v.repaint_delay.is_zero()) {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    #[cfg(feature = "dhat")]
    let _profiler = dhat::Profiler::new_heap();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    mem_log("main start");

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: svgwatch <file.svg>")?;
    if !path.exists() {
        anyhow::bail!("SVG file not found: {}", path.display());
    }
    // Canonicalize so `parent()` is always a real directory and watcher
    // event paths can be compared by equality.
    let path = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;

    let event_loop = EventLoop::<AppEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let mut app = App::new(path, proxy)?;

    event_loop.run_app(&mut app)?;
    Ok(())
}
