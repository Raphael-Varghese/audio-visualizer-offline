import sys
import math
import numpy as np
import sounddevice as sd
from collections import deque
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
import pyqtgraph as pg

# ---------------------------------------------------------------------------
# Audio Capture Thread
# ---------------------------------------------------------------------------

class AudioCaptureThread(QThread):
    data_ready = pyqtSignal(np.ndarray, np.ndarray, float, float, float)

    def __init__(self, samplerate=48000, blocksize=2048):
        super().__init__()
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = 2
        self.running = True
        self.device = None
        self.use_wasapi_loopback = False
        self._find_loopback_device()

    def _find_loopback_device(self):
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        print("\n[Audio] Scanning audio devices...")
        for i, dev in enumerate(devices):
            api_name = hostapis[dev['hostapi']]['name']
            print(f"  [{i}] {dev['name']} | in={dev['max_input_channels']} out={dev['max_output_channels']} | {api_name}")

        for i, dev in enumerate(devices):
            if 'stereo mix' in dev['name'].lower() and dev['max_input_channels'] > 0:
                self.device = i
                self.channels = min(2, dev['max_input_channels'])
                print(f"[Audio] Using Stereo Mix: {dev['name']} (index {i}, {self.channels}ch)")
                return

        keywords = ['what u hear', 'wave out', 'monitor', 'loopback']
        for kw in keywords:
            for i, dev in enumerate(devices):
                if kw in dev['name'].lower() and dev['max_input_channels'] > 0:
                    self.device = i
                    self.channels = min(2, dev['max_input_channels'])
                    print(f"[Audio] Using loopback: {dev['name']} (index {i}, {self.channels}ch)")
                    return

        wasapi_idx = None
        for i, api in enumerate(hostapis):
            if 'wasapi' in api['name'].lower():
                wasapi_idx = i
                break

        if wasapi_idx is not None:
            default_out = hostapis[wasapi_idx]['default_output_device']
            dev = devices[default_out]
            self.device = default_out
            self.use_wasapi_loopback = True
            self.channels = 2
            print(f"[Audio] WASAPI loopback: {dev['name']} (index {default_out}, loopback=True)")
            return

        for i, dev in enumerate(devices):
            if 'monitor' in dev['name'].lower() and dev['max_input_channels'] > 0:
                self.device = i
                self.channels = min(2, dev['max_input_channels'])
                print(f"[Audio] Monitor: {dev['name']} (index {i})")
                return

        try:
            default_in = sd.query_devices(kind='input')
            self.device = default_in['index']
            self.channels = min(2, default_in['max_input_channels'])
            print(f"[Audio] Fallback mic: {default_in['name']} (index {self.device})")
            print("[Audio] WARNING: Capturing microphone, not desktop audio!")
        except Exception as e:
            print(f"[Audio] No input device: {e}")

    def run(self):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio] Status: {status}")
            if not self.running:
                raise sd.CallbackStop

            audio = indata.astype(np.float64)
            if audio.shape[1] > 1:
                mono = np.mean(audio, axis=1)
                left = audio[:, 0]
                right = audio[:, 1]
            else:
                mono = audio[:, 0]
                left = right = mono

            window = np.hanning(len(mono))
            windowed = mono * window
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)

            def dbfs(samples):
                peak = np.max(np.abs(samples))
                if peak < 1e-10:
                    return -120.0
                return 20 * math.log10(peak)

            left_db = dbfs(left)
            right_db = dbfs(right)

            if audio.shape[1] > 1:
                left_std = np.std(left)
                right_std = np.std(right)
                if left_std < 1e-12 or right_std < 1e-12:
                    corr = 0.0
                else:
                    corr = np.corrcoef(left, right)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
            else:
                corr = 1.0

            self.data_ready.emit(magnitude, mono, left_db, right_db, corr)

        try:
            stream_kwargs = dict(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=np.float32,
                callback=callback,
            )
            if self.use_wasapi_loopback:
                try:
                    stream_kwargs['extra_settings'] = sd.WasapiSettings(loopback=True)
                    print("[Audio] WASAPI loopback enabled")
                except AttributeError:
                    pass

            with sd.InputStream(**stream_kwargs):
                print(f"[Audio] Stream started: sr={self.samplerate}, block={self.blocksize}, dev={self.device}, ch={self.channels}")
                while self.running:
                    self.msleep(10)
        except Exception as e:
            print(f"[Audio] Stream error: {e}")
            dummy_fft = np.zeros(self.blocksize // 2 + 1)
            dummy_wave = np.zeros(self.blocksize)
            while self.running:
                self.data_ready.emit(dummy_fft, dummy_wave, -120.0, -120.0, 0.0)
                self.msleep(50)

    def stop(self):
        self.running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# Custom Meter Widgets
# ---------------------------------------------------------------------------

class BarMeter(QtWidgets.QWidget):
    def __init__(self, label="", parent=None):
        super().__init__(parent)
        self.label = label
        self.value_db = -120.0
        self.peak_db = -120.0
        self.peak_fall_timer = 0
        self.setMinimumSize(50, 300)

    def set_value(self, db):
        self.value_db = db
        if db > self.peak_db:
            self.peak_db = db
            self.peak_fall_timer = 50
        else:
            self.peak_fall_timer -= 1
            if self.peak_fall_timer <= 0:
                self.peak_db -= 1.0
                self.peak_fall_timer = 2
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 6
        bar_w = w - pad * 2
        bar_h = h - pad * 2 - 20

        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 45), 1))
        painter.setBrush(QtGui.QColor(14, 14, 18))
        painter.drawRoundedRect(pad, pad, bar_w, bar_h, 4, 4)

        min_db, max_db = -60.0, 6.0
        t = (self.value_db - min_db) / (max_db - min_db)
        t = max(0.0, min(1.0, t))
        fill_h = int(bar_h * t)

        grad = QtGui.QLinearGradient(pad, pad + bar_h, pad, pad)
        grad.setColorAt(0.0, QtGui.QColor(0, 220, 0))
        grad.setColorAt(0.45, QtGui.QColor(230, 240, 0))
        grad.setColorAt(0.75, QtGui.QColor(255, 120, 0))
        grad.setColorAt(1.0, QtGui.QColor(255, 30, 50))
        painter.setBrush(grad)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pad, pad + bar_h - fill_h, bar_w, fill_h, 4, 4)

        peak_t = (self.peak_db - min_db) / (max_db - min_db)
        peak_t = max(0.0, min(1.0, peak_t))
        peak_y = pad + bar_h - int(bar_h * peak_t)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        painter.drawLine(pad, peak_y, pad + bar_w, peak_y)

        painter.setPen(QtGui.QPen(QtGui.QColor(80, 80, 90), 1))
        for db_val in [-60, -40, -20, -6, 0]:
            tick_t = (db_val - min_db) / (max_db - min_db)
            tick_y = pad + bar_h - int(bar_h * tick_t)
            painter.drawLine(pad + bar_w - 8, tick_y, pad + bar_w, tick_y)

        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))
        painter.drawText(QtCore.QRect(0, h - 18, w, 18), Qt.AlignmentFlag.AlignCenter, self.label)
        painter.end()


class CircularMeter(QtWidgets.QWidget):
    def __init__(self, label="", min_val=-1.0, max_val=1.0, parent=None):
        super().__init__(parent)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = 0.0
        self.setMinimumSize(160, 160)

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        size = min(w, h) - 16
        rect = QtCore.QRectF((w - size) / 2, (h - size) / 2, size, size)

        pen_bg = QtGui.QPen(QtGui.QColor(45, 45, 55), size * 0.14)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        t = max(0.0, min(1.0, t))
        angle = int(-270 * 16 * t)
        color = QtGui.QColor(0, 230, 130) if abs(self.value) > 0.5 else QtGui.QColor(230, 190, 0)
        pen_val = QtGui.QPen(color, size * 0.14)
        pen_val.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_val)
        painter.drawArc(rect, 225 * 16, angle)

        painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240)))
        font = painter.font()
        font.setPointSize(max(10, int(size / 7)))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.2f}")
        painter.drawText(QtCore.QRectF(0, h - 26, w, 26), Qt.AlignmentFlag.AlignCenter, self.label)
        painter.end()


class SpectrumBarsWidget(QtWidgets.QWidget):
    def __init__(self, num_bands=31, fun_mode=False, parent=None):
        super().__init__(parent)
        self.num_bands = num_bands
        self.fun_mode = fun_mode
        self.bands = np.zeros(num_bands)
        self.peaks = np.zeros(num_bands)
        self.fall = np.ones(num_bands) * 2.0
        self.setMinimumSize(700, 360)
        self._timer = 0

    def set_bands(self, values):
        self.bands = values
        self._timer += 1
        for i, v in enumerate(values):
            if v > self.peaks[i]:
                self.peaks[i] = v
                self.fall[i] = 6.0
            else:
                self.fall[i] = max(0.4, self.fall[i] * 0.88)
                self.peaks[i] -= self.fall[i]
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 12
        gap = 2
        bar_w = (w - pad * 2 - (self.num_bands - 1) * gap) / self.num_bands

        for i in range(self.num_bands):
            x = pad + i * (bar_w + gap)
            val = max(0.0, min(1.0, self.bands[i]))
            peak = max(0.0, min(1.0, self.peaks[i]))
            bar_h = val * (h - pad * 2 - 14)
            peak_y = h - pad - 14 - peak * (h - pad * 2 - 14)

            if self.fun_mode:
                hue = (i / self.num_bands) * 300 + self._timer * 3
                color = QtGui.QColor.fromHsv(int(hue) % 360, 250, 255)
                peak_color = QtGui.QColor(255, 255, 255)
            else:
                if val < 0.45:
                    color = QtGui.QColor(0, 220 + int(val * 80), 0)
                elif val < 0.75:
                    color = QtGui.QColor(240, 240, 0)
                else:
                    color = QtGui.QColor(255, 40 + int((1 - val) * 60), 0)
                peak_color = QtGui.QColor(255, 255, 255)

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            bw = max(1, int(bar_w))
            painter.drawRoundedRect(int(x), int(h - pad - 14 - bar_h), bw, int(bar_h), 3, 3)

            painter.setPen(QtGui.QPen(peak_color, 2))
            painter.drawLine(int(x), int(peak_y), int(x + bw), int(peak_y))

        painter.end()


class WaveformWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = np.zeros(2048)
        self.setMinimumSize(400, 160)

    def set_samples(self, samples):
        self.samples = samples
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h // 2
        painter.fillRect(self.rect(), QtGui.QColor(10, 10, 15))
        pen = QtGui.QPen(QtGui.QColor(0, 255, 150), 1.5)
        painter.setPen(pen)

        n = len(self.samples)
        if n < 2:
            return
        step = max(1, n // w)
        points = []
        for x in range(0, min(w, n // step)):
            idx = x * step
            y = mid - self.samples[idx] * (h * 0.45)
            points.append(QtCore.QPointF(x, y))
        if len(points) > 1:
            painter.drawPolyline(points)
        painter.end()


class SpectrogramWidget(pg.PlotWidget):
    def __init__(self, freqs, max_time_frames=400, parent=None):
        super().__init__(parent)
        self.all_freqs = freqs
        self.max_time_frames = max_time_frames
        self.display_bins = 400

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', 'Time')
        self.getAxis('left').setTextPen('w')
        self.getAxis('bottom').setTextPen('w')
        self.setMinimumSize(440, 220)

        step = max(1, len(freqs) // self.display_bins)
        self.display_freqs = freqs[::step][:self.display_bins]
        self._max_freq = float(self.display_freqs[-1]) if len(self.display_freqs) > 0 else 20000.0
        self._freq_step = self.display_freqs[1] - self.display_freqs[0] if len(self.display_freqs) > 1 else 50.0

        colors = [
            (0, 0, 0),
            (40, 0, 60),
            (130, 0, 90),
            (230, 0, 60),
            (255, 100, 0),
            (255, 220, 0),
            (255, 255, 255),
        ]
        positions = [0.0, 0.12, 0.30, 0.50, 0.72, 0.90, 1.0]
        cmap = pg.ColorMap(positions, [pg.mkColor(c) for c in colors])
        self.img.setColorMap(cmap)

        self.history = deque(maxlen=max_time_frames)

    def push_spectrum(self, full_spectrum_db):
        step = max(1, len(full_spectrum_db) // self.display_bins)
        spectrum = full_spectrum_db[::step][:self.display_bins]
        self.history.append(spectrum)

        if len(self.history) < 2:
            return

        data = np.array(self.history)
        data = data.T
        data = np.flipud(data)

        self.img.setImage(data, autoLevels=False)
        self.img.setLevels([-82, -28])

        time_frames = len(self.history)
        self.img.setRect(0, 0, time_frames, self._max_freq)

        if time_frames == 2:
            self.autoRange()


# ---------------------------------------------------------------------------
# Particle Visualizer  -- with BASS TRANSIENT DETECTION
# ---------------------------------------------------------------------------

class ParticleVisualizer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.particles = []
        self.bass_level = 0.0
        self._display_bass = 0.0
        self._debug_text = "Transient: 0.000 | Particles: 0"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_particles)
        self._timer.start(16)
        self.setMinimumSize(440, 340)

    def set_bass(self, transient_norm, overall_db=-120.0):
        # SILENCE GATE
        if overall_db < -50.0:
            self.bass_level = 0.0
        else:
            self.bass_level = transient_norm
        self._overall_db = overall_db

    def _update_particles(self):
        # Smooth interpolation
        diff = self.bass_level - self._display_bass
        if diff > 0:
            self._display_bass += diff * 0.25
        else:
            self._display_bass += diff * 0.20
        self._display_bass = max(0.0, self._display_bass)
        db = self._display_bass

        # RAISED threshold: only strong transients spawn
        if db > 0.35 and len(self.particles) < 500:
            for _ in range(int(db * 14)):
                self.particles.append({
                    'x': self.width() / 2 + np.random.normal(0, 60),
                    'y': self.height() / 2 + np.random.normal(0, 60),
                    'vx': np.random.normal(0, db * 16),
                    'vy': np.random.normal(-db * 18, 7),
                    'life': 1.0,
                    'hue': np.random.randint(0, 360),
                    'size': np.random.uniform(2, 10),
                })

        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.14
            p['life'] -= 0.010

        self.particles = [p for p in self.particles if p['life'] > 0]

        self._debug_text = (
            f"Transient: {self.bass_level:.3f} | "
            f"Overall: {getattr(self, '_overall_db', -120.0):.1f} dB | "
            f"Particles: {len(self.particles)}"
        )
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(5, 5, 10))

        for p in self.particles:
            color = QtGui.QColor.fromHsv(p['hue'], 240, int(255 * p['life']))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QtCore.QPointF(p['x'], p['y']),
                p['size'] * p['life'],
                p['size'] * p['life']
            )

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 128)))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(QtCore.QRect(10, 10, self.width() - 20, 20),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                        self._debug_text)
        painter.end()


# ---------------------------------------------------------------------------
# SMOOTHED Circular Spectrum Widget
# ---------------------------------------------------------------------------

class CircularSpectrumWidget(QtWidgets.QWidget):
    def __init__(self, num_bands=64, parent=None):
        super().__init__(parent)
        self.num_bands = num_bands
        self.bands = np.zeros(num_bands)
        self._display_bands = np.zeros(num_bands)
        self.setMinimumSize(240, 240)

    def set_bands(self, values):
        if len(values) >= self.num_bands:
            self.bands = values[:self.num_bands]
        else:
            self.bands[:len(values)] = values
        self.update()

    def paintEvent(self, event):
        self._display_bands += (self.bands - self._display_bands) * 0.15

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.14
        painter.fillRect(self.rect(), QtGui.QColor(5, 5, 10))

        for i in range(self.num_bands):
            angle = 2 * math.pi * i / self.num_bands - math.pi / 2
            val = max(0.0, min(1.0, self._display_bands[i]))
            bar_len = val * radius * 4.5
            hue = (i / self.num_bands) * 360
            color = QtGui.QColor.fromHsv(int(hue), 250, 255)
            pen = QtGui.QPen(color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x1 = cx + math.cos(angle) * radius
            y1 = cy + math.sin(angle) * radius
            x2 = cx + math.cos(angle) * (radius + bar_len)
            y2 = cy + math.sin(angle) * (radius + bar_len)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        glow = QtGui.QRadialGradient(cx, cy, radius * 0.55)
        glow.setColorAt(0.0, QtGui.QColor(255, 255, 255, 120))
        glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.setBrush(glow)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(QtCore.QPointF(cx, cy), radius * 0.55, radius * 0.55)
        painter.end()


# ---------------------------------------------------------------------------
# Main Window  -- FUNK EDITION with BASS TRANSIENT DETECTION
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphical Audio Analyzer")
        self.resize(1360, 950)
        self.setStyleSheet("background-color: #0a0a10;")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                background: #14141e;
                color: #888;
                padding: 14px 40px;
                font-weight: bold;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: #242434;
                color: #fff;
            }
        """)
        layout.addWidget(self.tabs)

        self.samplerate = 48000
        self.blocksize = 2048
        self.freqs = np.fft.rfftfreq(self.blocksize, 1 / self.samplerate)

        self.funk_weights = self._funk_weights(self.freqs)

        self.smoothed_tech = np.zeros(31)
        self.smoothed_fun = np.zeros(64)
        self._fun_history = deque(maxlen=4)
        self._fun_display = np.zeros(64)

        self.attack = 0.70
        self.decay = 0.75
        self.fun_attack = 0.60
        self.fun_decay = 0.72

        # ---- BASS TRANSIENT DETECTION state ----
        self.bass_env_fast = 0.0
        self.bass_env_slow = 0.0
        self.bass_transient_history = deque(maxlen=60)

        self._build_technical_tab()
        self._build_fun_tab()

        self.audio_thread = AudioCaptureThread(self.samplerate, self.blocksize)
        self.audio_thread.data_ready.connect(self._on_audio_data)
        self.audio_thread.start()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(16)

        self._latest_fft = None
        self._latest_wave = None
        self._latest_left_db = -120.0
        self._latest_right_db = -120.0
        self._latest_corr = 0.0

    def _funk_weights(self, freqs):
        weights = np.ones_like(freqs, dtype=np.float64)
        for i, f in enumerate(freqs):
            w = 1.0
            if f < 40:
                w = 0.5
            elif 40 <= f < 60:
                w = 1.3
            elif 60 <= f < 120:
                w = 2.5
            elif 120 <= f < 250:
                w = 2.0
            elif 250 <= f < 600:
                w = 1.8
            elif 600 <= f < 1500:
                w = 1.4
            elif 1500 <= f < 4000:
                w = 1.6
            elif 4000 <= f < 8000:
                w = 1.3
            elif 8000 <= f < 12000:
                w = 1.1
            elif f >= 14000:
                w = 0.6
            weights[i] = w
        return weights

    def _build_technical_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        grid.setSpacing(18)
        grid.setContentsMargins(22, 22, 22, 22)

        self.tech_spectrum = SpectrumBarsWidget(num_bands=31, fun_mode=False)
        grid.addWidget(self._frame("31-Band Spectrum Analyzer", self.tech_spectrum), 0, 0, 1, 4)

        meter_box = QtWidgets.QHBoxLayout()
        self.left_meter = BarMeter("L")
        self.right_meter = BarMeter("R")
        self.loudness_meter = BarMeter("LUFS ≈")
        self.corr_meter = CircularMeter("Correlation", -1.0, 1.0)
        meter_box.addWidget(self._frame("Left", self.left_meter))
        meter_box.addWidget(self._frame("Right", self.right_meter))
        meter_box.addWidget(self._frame("Loudness", self.loudness_meter))
        meter_box.addWidget(self._frame("Phase", self.corr_meter))
        grid.addLayout(meter_box, 1, 0, 1, 4)

        self.waveform = WaveformWidget()
        grid.addWidget(self._frame("Waveform / Oscilloscope", self.waveform), 2, 0, 1, 2)

        self.spectrogram = SpectrogramWidget(self.freqs)
        grid.addWidget(self._frame("Spectrogram", self.spectrogram), 2, 2, 1, 2)
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px; padding: 6px;")
        grid.addWidget(info, 3, 0, 1, 4)

        self.tabs.addTab(tab, "Technical Mode")

    def _build_fun_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        vbox.setSpacing(18)
        vbox.setContentsMargins(22, 22, 22, 22)

        self.fun_spectrum = SpectrumBarsWidget(num_bands=64, fun_mode=True)
        vbox.addWidget(self._frame("Neon Spectrum (64-band)", self.fun_spectrum))

        hbox = QtWidgets.QHBoxLayout()
        self.circular_spectrum = CircularSpectrumWidget(num_bands=64)
        self.particles = ParticleVisualizer()
        hbox.addWidget(self._frame("Circular Spectrum", self.circular_spectrum))
        hbox.addWidget(self._frame("Bass Particles", self.particles))
        vbox.addLayout(hbox)

        self.tabs.addTab(tab, "Fun Mode")

    def _frame(self, title, widget):
        frame = QtWidgets.QGroupBox(title)
        frame.setStyleSheet("""
            QGroupBox {
                color: #aab;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #2a2a3a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.addWidget(widget)
        return frame

    def _on_audio_data(self, fft_mag, waveform, left_db, right_db, corr):
        self._latest_fft = fft_mag
        self._latest_wave = waveform
        self._latest_left_db = left_db
        self._latest_right_db = right_db
        self._latest_corr = corr

    def _update_ui(self):
        if self._latest_fft is None or self._latest_wave is None:
            return

        fft = self._latest_fft
        wave = self._latest_wave

        # ---- Spectrum bars: per-frame normalized, funk-weighted ----
        weighted_fft = fft * self.funk_weights
        frame_peak = np.max(weighted_fft)
        normalized = weighted_fft / (frame_peak + 1e-12)
        fft_db = 20 * np.log10(normalized + 1e-12)
        fft_db = np.clip(fft_db, -90, 0)

        # Technical spectrum
        raw_tech = self._log_bands(fft_db, self.freqs, 31, 20, 20000)
        for i in range(31):
            if raw_tech[i] > self.smoothed_tech[i]:
                self.smoothed_tech[i] = self.smoothed_tech[i] * (1 - self.attack) + raw_tech[i] * self.attack
            else:
                self.smoothed_tech[i] *= self.decay
        self.tech_spectrum.set_bands(self.smoothed_tech)

        # Smooth fun spectrum
        raw_fun = self._linear_bands(fft_db, self.freqs, 64, 0, 10000)
        for i in range(64):
            if raw_fun[i] > self.smoothed_fun[i]:
                self.smoothed_fun[i] = self.smoothed_fun[i] * (1 - self.fun_attack) + raw_fun[i] * self.fun_attack
            else:
                self.smoothed_fun[i] *= self.fun_decay

        self._fun_history.append(self.smoothed_fun.copy())
        if len(self._fun_history) >= 2:
            avg = np.mean(self._fun_history, axis=0)
            self._fun_display += (avg - self._fun_display) * 0.30
        else:
            self._fun_display = self.smoothed_fun.copy()

        self.fun_spectrum.set_bands(self._fun_display)
        self.circular_spectrum.set_bands(self._fun_display)

        # Meters
        self.left_meter.set_value(self._latest_left_db)
        self.right_meter.set_value(self._latest_right_db)
        loudness = max(self._latest_left_db, self._latest_right_db)
        self.loudness_meter.set_value(loudness)
        self.corr_meter.set_value(self._latest_corr)

        # Waveform
        self.waveform.set_samples(wave)

        # Spectrogram
        weighted_db = 20 * np.log10(weighted_fft / (np.max(weighted_fft) + 1e-12) + 1e-12)
        weighted_db = np.clip(weighted_db, -120, 0)
        self.spectrogram.push_spectrum(weighted_db)

        # =====================================================================
        # BASS TRANSIENT DETECTION (not normalized - uses raw FFT envelopes)
        # =====================================================================
        # Use RAW fft (not weighted, not normalized) for transient detection
        # This detects actual kick drum ONSETS, not sustained bass
        bass_mask = (self.freqs >= 50) & (self.freqs <= 140)
        bass_mag = np.mean(fft[bass_mask])  # raw magnitude, no weighting

        # Fast envelope: follows transients (kick drum attacks)
        if bass_mag > self.bass_env_fast:
            self.bass_env_fast += (bass_mag - self.bass_env_fast) * 0.65
        else:
            self.bass_env_fast *= 0.78

        # Slow envelope: follows sustained bass level (baseline)
        if bass_mag > self.bass_env_slow:
            self.bass_env_slow += (bass_mag - self.bass_env_slow) * 0.03
        else:
            self.bass_env_slow *= 0.992

        # Transient = spike above the sustained baseline
        transient = self.bass_env_fast - self.bass_env_slow * 1.15
        transient = max(0.0, transient)

        # Normalize against recent peak transient
        self.bass_transient_history.append(transient)
        recent_peak = np.max(self.bass_transient_history) if len(self.bass_transient_history) > 5 else 1e-10
        transient_norm = min(1.0, transient / (recent_peak + 1e-12))

        overall_db = max(self._latest_left_db, self._latest_right_db)
        self.particles.set_bass(transient_norm, overall_db)

    def _log_bands(self, fft_db, freqs, n_bands, f_min, f_max):
        bands = np.zeros(n_bands)
        log_min, log_max = math.log10(max(f_min, 1)), math.log10(f_max)
        for i in range(n_bands):
            f0 = 10 ** (log_min + (log_max - log_min) * i / n_bands)
            f1 = 10 ** (log_min + (log_max - log_min) * (i + 1) / n_bands)
            mask = (freqs >= f0) & (freqs < f1)
            if np.any(mask):
                db = np.max(fft_db[mask])
                bands[i] = (db + 90) / 90
            else:
                bands[i] = 0.0
        return np.clip(bands, 0, 1)

    def _linear_bands(self, fft_db, freqs, n_bands, f_min, f_max):
        bands = np.zeros(n_bands)
        step = (f_max - f_min) / n_bands
        for i in range(n_bands):
            f0 = f_min + i * step
            f1 = f0 + step
            mask = (freqs >= f0) & (freqs < f1)
            if np.any(mask):
                db = np.max(fft_db[mask])
                bands[i] = (db + 90) / 90
            else:
                bands[i] = 0.0
        return np.clip(bands, 0, 1)

    def closeEvent(self, event):
        self.audio_thread.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(10, 10, 15))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(14, 14, 20))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(24, 24, 36))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(24, 24, 36))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(240, 240, 240))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
