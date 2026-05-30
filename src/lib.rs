//! # Spectral Emotions
//!
//! An experimental crate that maps emotional states to spectral properties —
//! the intersection of psychology and graph theory.
//!
//! Emotions have a "spectrum" just like signals. We can detect emotional states
//! from the spectral properties of a vibe signal.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Plutchik's 8 primary emotions with metaphorical spectral frequencies.
// Frequencies are spread across a metaphorical "emotional spectrum" (1–8 Hz).
// ---------------------------------------------------------------------------

/// Plutchik's eight primary emotions, each with a metaphorical spectral frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Emotion {
    Joy,
    Trust,
    Fear,
    Surprise,
    Sadness,
    Disgust,
    Anger,
    Anticipation,
}

impl Emotion {
    /// Metaphorical spectral frequency (Hz-like) for this emotion.
    pub const fn frequency(self) -> f64 {
        match self {
            Self::Joy => 1.0,
            Self::Trust => 2.0,
            Self::Fear => 3.0,
            Self::Surprise => 4.0,
            Self::Sadness => 5.0,
            Self::Disgust => 6.0,
            Self::Anger => 7.0,
            Self::Anticipation => 8.0,
        }
    }

    /// Create an [`EmotionState`] with the given intensity (clamped to 0–1).
    pub fn intensity(self, intensity: f64) -> EmotionState {
        EmotionState {
            emotion: self,
            intensity: intensity.clamp(0.0, 1.0),
            timestamp: 0,
        }
    }

    /// All eight primary emotions in order.
    pub const fn all() -> &'static [Emotion; 8] {
        const EMOTIONS: [Emotion; 8] = [
            Emotion::Joy,
            Emotion::Trust,
            Emotion::Fear,
            Emotion::Surprise,
            Emotion::Sadness,
            Emotion::Disgust,
            Emotion::Anger,
            Emotion::Anticipation,
        ];
        &EMOTIONS
    }
}

// ---------------------------------------------------------------------------
// EmotionState
// ---------------------------------------------------------------------------

/// A single emotional reading: which emotion, how strong, and when.
#[derive(Debug, Clone, Copy)]
pub struct EmotionState {
    pub emotion: Emotion,
    pub intensity: f64,
    pub timestamp: u64,
}

impl EmotionState {
    /// Create a new `EmotionState` with an explicit timestamp.
    pub fn new(emotion: Emotion, intensity: f64, timestamp: u64) -> Self {
        Self {
            emotion,
            intensity: intensity.clamp(0.0, 1.0),
            timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// EmotionSpectrum — the "FFT of feelings"
// ---------------------------------------------------------------------------

/// Maps all 8 emotions to intensities, like a spectral decomposition of an
/// emotional signal.
#[derive(Debug, Clone)]
pub struct EmotionSpectrum {
    values: [f64; 8],
}

impl EmotionSpectrum {
    /// Create an empty (all-zero) spectrum.
    pub fn empty() -> Self {
        Self { values: [0.0; 8] }
    }

    /// Create a uniform spectrum with every emotion at the same intensity.
    pub fn uniform(intensity: f64) -> Self {
        Self {
            values: [intensity; 8],
        }
    }

    /// Get the intensity of a specific emotion.
    pub fn get(&self, emotion: Emotion) -> f64 {
        self.values[emotion as usize]
    }

    /// Set the intensity of a specific emotion.
    pub fn set(&mut self, emotion: Emotion, intensity: f64) {
        self.values[emotion as usize] = intensity.clamp(0.0, 1.0);
    }

    /// Derive an emotion spectrum from a "vibe signal" — a time-series of f64
    /// values. Each emotion's frequency is used as a basis function; we project
    /// the signal onto each frequency component (simplified DFT).
    pub fn from_vibe_history(history: &[f64]) -> Self {
        let mut spectrum = Self::empty();
        if history.is_empty() {
            return spectrum;
        }

        let n = history.len() as f64;
        for (i, emotion) in Emotion::all().iter().enumerate() {
            let freq = emotion.frequency();
            let mut real = 0.0;
            let mut imag = 0.0;
            for (t, val) in history.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * freq * (t as f64) / n;
                real += val * angle.cos();
                imag += val * angle.sin();
            }
            let magnitude = (real * real + imag * imag).sqrt() / n;
            spectrum.values[i] = magnitude.clamp(0.0, 1.0);
        }
        spectrum
    }

    /// Return the emotion with the highest intensity.
    pub fn dominant_emotion(&self) -> Emotion {
        self.values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| Emotion::all()[i])
            .unwrap_or(Emotion::Joy)
    }

    /// Shannon entropy of the emotion distribution — measures emotional
    /// complexity. Higher values mean a more complex / diverse emotional state.
    pub fn emotional_complexity(&self) -> f64 {
        let total: f64 = self.values.iter().copied().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let mut entropy = 0.0;
        for &v in &self.values {
            if v > 0.0 {
                let p = v / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Blend two spectra together. `factor` = 0.0 gives `self`, 1.0 gives
    /// `other`, 0.5 is an equal mix.
    pub fn blend(&self, other: &EmotionSpectrum, factor: f64) -> EmotionSpectrum {
        let f = factor.clamp(0.0, 1.0);
        let mut result = Self::empty();
        for i in 0..8 {
            result.values[i] = self.values[i] * (1.0 - f) + other.values[i] * f;
        }
        result
    }

    /// Map the emotion spectrum to an RGB colour. Joy→yellow, Sadness→blue,
    /// Anger→red, etc. The contribution of each emotion is weighted by its
    /// intensity.
    pub fn to_color(&self) -> [f64; 3] {
        // Each emotion has an RGB contribution.
        let palette: [[f64; 3]; 8] = [
            [1.0, 1.0, 0.0], // Joy — yellow
            [0.0, 0.8, 0.2], // Trust — green
            [0.3, 0.0, 0.5], // Fear — dark purple
            [1.0, 0.5, 0.0], // Surprise — orange
            [0.0, 0.2, 1.0], // Sadness — blue
            [0.5, 0.5, 0.0], // Disgust — olive
            [1.0, 0.0, 0.0], // Anger — red
            [0.9, 0.9, 1.0], // Anticipation — light lavender
        ];

        let mut color = [0.0_f64; 3];
        for (i, &v) in self.values.iter().enumerate() {
            color[0] += palette[i][0] * v;
            color[1] += palette[i][1] * v;
            color[2] += palette[i][2] * v;
        }

        // Normalize so the brightest channel is ≤ 1.0.
        let max = color.iter().copied().fold(0.0_f64, f64::max);
        if max > 1.0 {
            color[0] /= max;
            color[1] /= max;
            color[2] /= max;
        }
        color
    }
}

// ---------------------------------------------------------------------------
// EmotionalGraph
// ---------------------------------------------------------------------------

/// A graph whose nodes carry emotion spectra, connected by emotional resonance
/// edges.
#[derive(Debug, Clone, Default)]
pub struct EmotionalGraph {
    nodes: HashMap<String, EmotionSpectrum>,
    edges: HashMap<(String, String), f64>,
}

impl EmotionalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node with its emotion spectrum.
    pub fn add_node(&mut self, id: impl Into<String>, spectrum: EmotionSpectrum) {
        self.nodes.insert(id.into(), spectrum);
    }

    /// Add an emotional-resonance edge between two nodes.
    /// Resonance is clamped to [0, 1].
    pub fn add_edge(&mut self, a: &str, b: &str, resonance: f64) {
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.edges.insert(key, resonance.clamp(0.0, 1.0));
    }

    /// Aggregate emotional state of the whole graph (weighted average of all
    /// node spectra).
    pub fn emotional_field(&self) -> EmotionSpectrum {
        if self.nodes.is_empty() {
            return EmotionSpectrum::empty();
        }
        let mut result = EmotionSpectrum::empty();
        let n = self.nodes.len() as f64;
        for spectrum in self.nodes.values() {
            for i in 0..8 {
                result.values[i] += spectrum.values[i];
            }
        }
        for v in &mut result.values {
            *v /= n;
        }
        result
    }

    /// How emotionally similar are two nodes? Returns 0.0 (identical) to a
    /// positive distance (very different). Returns `f64::INFINITY` if either
    /// node is missing.
    pub fn empathy_distance(&self, a: &str, b: &str) -> f64 {
        let sa = match self.nodes.get(a) {
            Some(s) => s,
            None => return f64::INFINITY,
        };
        let sb = match self.nodes.get(b) {
            Some(s) => s,
            None => return f64::INFINITY,
        };
        let mut dist = 0.0;
        for i in 0..8 {
            let d = sa.values[i] - sb.values[i];
            dist += d * d;
        }
        dist.sqrt()
    }
}

// ---------------------------------------------------------------------------
// EmotionToVoxel — convert emotions to voxel-world properties
// ---------------------------------------------------------------------------

/// Weather types for the voxel world, driven by emotional state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weather {
    Sunny,
    Rainy,
    Stormy,
    Snowy,
    Aurora,
    Foggy,
    MeteorShower,
}

/// Converts emotions to voxel-world properties: materials, scales, and weather.
pub struct EmotionToVoxel;

impl EmotionToVoxel {
    /// Map an emotion to a voxel material type.
    pub fn emotion_to_material(e: &Emotion) -> String {
        match e {
            Emotion::Joy => "glow".into(),
            Emotion::Trust => "stone".into(),
            Emotion::Fear => "shadow".into(),
            Emotion::Surprise => "crystal".into(),
            Emotion::Sadness => "water".into(),
            Emotion::Disgust => "slime".into(),
            Emotion::Anger => "lava".into(),
            Emotion::Anticipation => "cloud".into(),
        }
    }

    /// Stronger emotion → bigger voxel structure. Returns a scale factor in
    /// [0.1, 10.0].
    pub fn intensity_to_scale(i: f64) -> f64 {
        0.1 + i.clamp(0.0, 1.0) * 9.9
    }

    /// Determine weather from the dominant emotion in a spectrum.
    pub fn spectrum_to_weather(s: &EmotionSpectrum) -> Weather {
        match s.dominant_emotion() {
            Emotion::Joy => Weather::Sunny,
            Emotion::Trust => Weather::Aurora,
            Emotion::Fear => Weather::Foggy,
            Emotion::Surprise => Weather::MeteorShower,
            Emotion::Sadness => Weather::Rainy,
            Emotion::Disgust => Weather::Foggy,
            Emotion::Anger => Weather::Stormy,
            Emotion::Anticipation => Weather::Snowy,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Emotion basics ----

    #[test]
    fn emotion_frequencies_are_distinct() {
        let freqs: Vec<f64> = Emotion::all().iter().map(|e| e.frequency()).collect();
        for i in 0..freqs.len() {
            for j in (i + 1)..freqs.len() {
                assert_ne!(freqs[i], freqs[j]);
            }
        }
    }

    #[test]
    fn emotion_frequencies_range_1_to_8() {
        for e in Emotion::all() {
            let f = e.frequency();
            assert!((1.0..=8.0).contains(&f));
        }
    }

    #[test]
    fn intensity_clamps_high() {
        let state = Emotion::Joy.intensity(5.0);
        assert!((state.intensity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn intensity_clamps_low() {
        let state = Emotion::Sadness.intensity(-0.5);
        assert!((state.intensity).abs() < 1e-9);
    }

    #[test]
    fn emotion_state_new_timestamp() {
        let state = EmotionState::new(Emotion::Anger, 0.7, 12345);
        assert_eq!(state.emotion, Emotion::Anger);
        assert!((state.intensity - 0.7).abs() < 1e-9);
        assert_eq!(state.timestamp, 12345);
    }

    // ---- EmotionSpectrum ----

    #[test]
    fn empty_spectrum_all_zeros() {
        let s = EmotionSpectrum::empty();
        for e in Emotion::all() {
            assert!((s.get(*e)).abs() < 1e-9);
        }
    }

    #[test]
    fn uniform_spectrum() {
        let s = EmotionSpectrum::uniform(0.5);
        for e in Emotion::all() {
            assert!((s.get(*e) - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn set_and_get() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Fear, 0.8);
        assert!((s.get(Emotion::Fear) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn from_vibe_history_empty() {
        let s = EmotionSpectrum::from_vibe_history(&[]);
        for e in Emotion::all() {
            assert!((s.get(*e)).abs() < 1e-9);
        }
    }

    #[test]
    fn from_vibe_history_single() {
        let s = EmotionSpectrum::from_vibe_history(&[1.0]);
        // With a single sample every emotion gets some magnitude.
        assert!(s.get(Emotion::Joy) > 0.0);
    }

    #[test]
    fn dominant_emotion_single() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Anger, 1.0);
        assert_eq!(s.dominant_emotion(), Emotion::Anger);
    }

    #[test]
    fn emotional_complexity_empty_is_zero() {
        let s = EmotionSpectrum::empty();
        assert!((s.emotional_complexity()).abs() < 1e-9);
    }

    #[test]
    fn emotional_complexity_uniform_is_max() {
        let s = EmotionSpectrum::uniform(0.5);
        let c = s.emotional_complexity();
        // Uniform distribution over 8 bins → log2(8) = 3.0
        assert!((c - 3.0).abs() < 1e-9);
    }

    #[test]
    fn blend_identity() {
        let s = EmotionSpectrum::uniform(0.5);
        let blended = s.blend(&s, 0.5);
        for e in Emotion::all() {
            assert!((blended.get(*e) - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn blend_zero_factor_is_self() {
        let mut a = EmotionSpectrum::empty();
        a.set(Emotion::Joy, 1.0);
        let b = EmotionSpectrum::uniform(0.5);
        let blended = a.blend(&b, 0.0);
        assert!((blended.get(Emotion::Joy) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn to_color_dominant_joy_is_yellowish() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Joy, 1.0);
        let rgb = s.to_color();
        assert!(rgb[0] > 0.9); // R
        assert!(rgb[1] > 0.9); // G
        assert!(rgb[2] < 0.1); // B — very little blue
    }

    #[test]
    fn to_color_dominant_anger_is_red() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Anger, 1.0);
        let rgb = s.to_color();
        assert!(rgb[0] > 0.9);
        assert!(rgb[1] < 0.1);
        assert!(rgb[2] < 0.1);
    }

    // ---- EmotionalGraph ----

    #[test]
    fn graph_empty_field() {
        let g = EmotionalGraph::new();
        let f = g.emotional_field();
        for e in Emotion::all() {
            assert!((f.get(*e)).abs() < 1e-9);
        }
    }

    #[test]
    fn graph_single_node_field() {
        let mut g = EmotionalGraph::new();
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Trust, 1.0);
        g.add_node("alice", s);
        let field = g.emotional_field();
        assert!((field.get(Emotion::Trust) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn graph_empathy_distance_identical() {
        let mut g = EmotionalGraph::new();
        let s = EmotionSpectrum::uniform(0.5);
        g.add_node("a", s.clone());
        g.add_node("b", s);
        let d = g.empathy_distance("a", "b");
        assert!(d.abs() < 1e-9);
    }

    #[test]
    fn graph_empathy_distance_missing() {
        let g = EmotionalGraph::new();
        assert!(g.empathy_distance("x", "y").is_infinite());
    }

    #[test]
    fn graph_empathy_distance_different() {
        let mut g = EmotionalGraph::new();
        let mut sa = EmotionSpectrum::empty();
        sa.set(Emotion::Joy, 1.0);
        let mut sb = EmotionSpectrum::empty();
        sb.set(Emotion::Sadness, 1.0);
        g.add_node("a", sa);
        g.add_node("b", sb);
        let d = g.empathy_distance("a", "b");
        assert!(d > 1.0);
    }

    // ---- EmotionToVoxel ----

    #[test]
    fn voxel_materials_all_distinct() {
        let materials: Vec<String> = Emotion::all()
            .iter()
            .map(|e| EmotionToVoxel::emotion_to_material(e))
            .collect();
        for i in 0..materials.len() {
            for j in (i + 1)..materials.len() {
                assert_ne!(materials[i], materials[j]);
            }
        }
    }

    #[test]
    fn voxel_material_joy_is_glow() {
        assert_eq!(EmotionToVoxel::emotion_to_material(&Emotion::Joy), "glow");
    }

    #[test]
    fn voxel_material_sadness_is_water() {
        assert_eq!(
            EmotionToVoxel::emotion_to_material(&Emotion::Sadness),
            "water"
        );
    }

    #[test]
    fn voxel_material_anger_is_lava() {
        assert_eq!(
            EmotionToVoxel::emotion_to_material(&Emotion::Anger),
            "lava"
        );
    }

    #[test]
    fn intensity_to_scale_bounds() {
        assert!((EmotionToVoxel::intensity_to_scale(0.0) - 0.1).abs() < 1e-9);
        assert!((EmotionToVoxel::intensity_to_scale(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn spectrum_weather_joy_sunny() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Joy, 1.0);
        assert_eq!(EmotionToVoxel::spectrum_to_weather(&s), Weather::Sunny);
    }

    #[test]
    fn spectrum_weather_sadness_rainy() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Sadness, 1.0);
        assert_eq!(EmotionToVoxel::spectrum_to_weather(&s), Weather::Rainy);
    }

    #[test]
    fn spectrum_weather_anger_stormy() {
        let mut s = EmotionSpectrum::empty();
        s.set(Emotion::Anger, 1.0);
        assert_eq!(EmotionToVoxel::spectrum_to_weather(&s), Weather::Stormy);
    }
}
