//! The splash: a spinning ASCII torus, after Andy Sloane's donut.c.
//!
//! [`frame`] is pure: the same angles and size always give the same rows.
//! The UI draws one under the banner while the chat is new and turns it a
//! little on every tick.

use std::f32::consts::TAU;

/// Characters from darkest to brightest, indexed by luminance.
pub const SHADES: &str = ".,-~:;=!*#$@";

/// Radius of the tube.
const R1: f32 = 1.0;
/// Distance from the torus's center to the middle of the tube.
const R2: f32 = 2.0;
/// Distance from the viewer to the torus's center.
const K2: f32 = 5.0;
/// Step around the tube's cross-section, at most.
const THETA_STEP: f32 = 0.07;
/// Step around the torus's axis, at most.
const PHI_STEP: f32 = 0.02;

/// One frame of the torus turned by `a` (about the x axis) and `b` (about
/// the z axis): `height` rows of `width` characters, centered, spaces where
/// there is nothing.
pub fn frame(a: f32, b: f32, width: u16, height: u16) -> Vec<String> {
    let (w, h) = (usize::from(width), usize::from(height));
    if w == 0 || h == 0 {
        return vec![String::new(); h];
    }
    let mut cells = vec![b' '; w * h];
    // 1/z of the nearest point drawn in each cell; 0 is infinitely far.
    let mut depth = vec![0.0_f32; w * h];

    // Terminal cells are about twice as tall as wide, so y is projected at
    // half scale. Turned, the torus spans at most 1.5·K1 columns and 0.75·K1
    // rows; K1 is about the largest scale that keeps it inside the frame.
    let k1 = f32::min(width as f32 / 1.6, height as f32 * 1.25);
    // Where the torus is nearest, one step moves up to 1.5·K1·dphi columns
    // (and 0.5·K1·dtheta); finer steps when it is big leave no gaps.
    let phi_step = PHI_STEP.min(0.6 / k1);
    let theta_step = THETA_STEP.min(1.6 / k1);
    let (sin_a, cos_a) = a.sin_cos();
    let (sin_b, cos_b) = b.sin_cos();

    let mut theta = 0.0_f32;
    while theta < TAU {
        let (sin_t, cos_t) = theta.sin_cos();
        // A point on the circle of the tube's cross-section.
        let circle_x = R2 + R1 * cos_t;
        let circle_y = R1 * sin_t;

        let mut phi = 0.0_f32;
        while phi < TAU {
            let (sin_p, cos_p) = phi.sin_cos();
            let x = circle_x * (cos_b * cos_p + sin_a * sin_b * sin_p) - circle_y * cos_a * sin_b;
            let y = circle_x * (sin_b * cos_p - sin_a * cos_b * sin_p) + circle_y * cos_a * cos_b;
            let ooz = 1.0 / (K2 + cos_a * circle_x * sin_p + circle_y * sin_a);
            let col = (w as f32 / 2.0 + k1 * ooz * x).floor();
            let row = (h as f32 / 2.0 - k1 / 2.0 * ooz * y).floor();
            if col >= 0.0 && row >= 0.0 && (col as usize) < w && (row as usize) < h {
                let index = row as usize * w + col as usize;
                if ooz > depth[index] {
                    depth[index] = ooz;
                    // Surface normal dotted with a light up and behind the viewer.
                    let luminance = cos_p * cos_t * sin_b - cos_a * cos_t * sin_p - sin_a * sin_t
                        + cos_b * (cos_a * sin_t - cos_t * sin_a * sin_p);
                    let shade = ((luminance * 8.0).max(0.0) as usize).min(SHADES.len() - 1);
                    cells[index] = SHADES.as_bytes()[shade];
                }
            }
            phi += phi_step;
        }
        theta += theta_step;
    }

    cells
        .chunks(w)
        .map(|row| row.iter().map(|&c| char::from(c)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_has_the_size_asked_for_and_draws_something() {
        let rows = frame(1.0, 1.0, 60, 22);
        assert_eq!(rows.len(), 22);
        assert!(rows.iter().all(|row| row.chars().count() == 60));
        assert!(rows.iter().any(|row| row.contains(|c| c != ' ')));
        assert!(rows.iter().any(|row| row.contains(['#', '$', '@'])));
    }

    #[test]
    fn an_unturned_torus_is_left_right_symmetric() {
        let rows = frame(0.0, 0.0, 60, 22);
        let drawn = |row: &String| {
            let first = row.find(|c| c != ' ')?;
            let last = row.rfind(|c| c != ' ')?;
            Some((first, last))
        };
        let (left, right) = rows
            .iter()
            .filter_map(drawn)
            .fold((usize::MAX, 0), |(l, r), (first, last)| {
                (l.min(first), r.max(last))
            });
        let margin_left = left;
        let margin_right = 59 - right;
        assert!(
            margin_left.abs_diff(margin_right) <= 2,
            "{margin_left} vs {margin_right}"
        );
    }

    #[test]
    fn turning_changes_the_frame() {
        assert_ne!(frame(0.0, 0.0, 60, 22), frame(0.7, 0.3, 60, 22));
        assert_eq!(frame(0.7, 0.3, 60, 22), frame(0.7, 0.3, 60, 22));
    }

    #[test]
    fn empty_sizes_do_not_panic() {
        assert!(frame(0.0, 0.0, 0, 0).is_empty());
        assert_eq!(frame(0.0, 0.0, 0, 3), vec![String::new(); 3]);
        assert!(frame(0.0, 0.0, 5, 0).is_empty());
    }
}
