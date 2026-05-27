const SHAPE_SMALL = [
  '      _..-===-.._',
  '    /(  )@@@@@(  )\\',
  '   /@@()@@@@@@@()@@\\',
  '  |@@(  )@@@@@(  )@@|',
  '  |@(    )@@@(    )@|',
  '  |(      )@(      )|',
  '   \\(    )@@@(    )/',
  '     .  )@@@@@(  .',
  '      \\)@@@@@@@(/',
  '       \\)@@@@@(/',
  '        .)@@@(.',
  '         \\---/',
  '          |.|',
  '         |---|',
  '          ===',
].join('\n');

const SHAPE_LARGE = [
  '              __....-==========-....__',
  '          _.-=(#######)      (#######)=-._',
  '       .-\'   (#########)    (#########)   \'-.',
  '      /##)  (###########)  (###########)  (##\\',
  '    ./####)(#############)(#############)(####\\.',
  '   /#####)  (###########)  (###########)  (#####\\',
  '  /#####)    (#########)    (#########)    (#####\\',
  ' |#####)      (#######)      (#######)      (#####|',
  ' |####)        (#####)        (#####)        (####|',
  ' |###)          (###)          (###)          (###|',
  ' |(#)            (#)            (#)            (#)|',
  ' |###)          (###)          (###)          (###|',
  ' |####)        (#####)        (#####)        (####|',
  ' |#####)      (#######)      (#######)      (#####|',
  '  \\#####)    (#########)    (#########)    (#####/',
  '   \\#####)  (###########)  (###########)  (#####/',
  '    \'\\####)(#############)(#############)(####/\'',
  '      \\##)  (###########)  (###########)  (##/',
  '       \\)    (#########)    (#########)    (/',
  '        \\     (#######)      (#######)     /',
  '         \'\\    (#####)        (#####)    /\'',
  '           \\    (###)          (###)    /',
  '            \\    (#)            (#)    /',
  '             \\  (###)          (###)  /',
  '              \'\\#####)        (#####/\'',
  '                \\#####)      (#####/',
  '                 \\#####)    (#####/',
  '                  \\#####)  (#####/',
  '                   \'\\####)(####/\'',
  '                     \\--------/',
  '                       ||  ||',
  '                       ||  ||',
  '                       ||  ||',
  '                     |---==---|',
  '                     |___..___|',
  '                     |___..___|',
  '                      ========',
].join('\n');

const SHAPE_TINY = [
  '      _..-===-.._',
  '    /)  (%%%%%)  (\\',
  '   /%%)(%%%%%%%)(%%\\',
  '  |%%)  (%%%%%)  (%%|',
  '  |%)    (%%%)    (%|',
  '  |)      (%)      (|',
  '   \\)    (%%%)    (/',
  '     .  (%%%%%)  .',
  '      \\(%%%%%%%)/',
  '       \\(%%%%%)/',
  '        .(%%%).',
  '         \\---/',
  '          |||',
  '         |===|',
  '         |___|',
].join('\n');

const SHAPES = {
  small: { art: SHAPE_SMALL, fillChar: '@', basketLines: 4 },
  large: { art: SHAPE_LARGE, fillChar: '#', basketLines: 8 },
  tiny:  { art: SHAPE_TINY,  fillChar: '%', basketLines: 4 },
};

export const BALLOON_PALETTES = {
  red:    { fill: '#93032E', outline: '#93032E', basket: '#93032E' },
  blue:   { fill: '#3F84E5', outline: '#3F84E5', basket: '#3F84E5' },
  orange: { fill: '#FE621D', outline: '#FE621D', basket: '#FE621D' },
};

export const BALLOON_CONFIG = {
  bobFraction: 0.18,
  bobPeriodMs: 4500,
  bobPeriodJitter: 0.3,
  atmStrength: 0.25,
  skyColor: '#FDF7EE',
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function hexToRgb(hex) {
  const normalized = hex.replace('#', '').trim();
  const full = normalized.length === 3
    ? normalized.split('').map((char) => char + char).join('')
    : normalized;
  return {
    r: parseInt(full.slice(0, 2), 16),
    g: parseInt(full.slice(2, 4), 16),
    b: parseInt(full.slice(4, 6), 16),
  };
}

function rgbToHex(r, g, b) {
  const toHex = (channel) => clamp(Math.round(channel), 0, 255)
    .toString(16)
    .padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function atmosphericTint(hex, amount, skyColor = BALLOON_CONFIG.skyColor) {
  const color = hexToRgb(hex);
  const sky = hexToRgb(skyColor);
  const mix = clamp(amount, 0, 1);
  return rgbToHex(
    color.r * (1 - mix) + sky.r * mix,
    color.g * (1 - mix) + sky.g * mix,
    color.b * (1 - mix) + sky.b * mix,
  );
}

export function renderBalloonHTML(art, fillChar, basketLines, palette, depthTint, skyColor) {
  const fillColor = atmosphericTint(palette.fill, depthTint, skyColor);
  const outlineColor = atmosphericTint(palette.outline, depthTint, skyColor);
  const basketColor = atmosphericTint(palette.basket, depthTint, skyColor);
  const lines = art.split('\n');
  const basketStart = lines.length - basketLines;
  let html = '';

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const inBasket = lineIndex >= basketStart;

    // Per-row silhouette: the opaque background spans from the first non-space
    // glyph to the last, so the diamond-shaped interior gaps stay opaque
    // (occluding birds behind the balloon body), while the exterior
    // whitespace around the row remains transparent.
    let firstNonSpace = -1;
    let lastNonSpace = -1;
    for (let i = 0; i < line.length; i += 1) {
      if (line[i] !== ' ') {
        if (firstNonSpace === -1) firstNonSpace = i;
        lastNonSpace = i;
      }
    }

    if (firstNonSpace === -1) {
      if (lineIndex < lines.length - 1) html += '\n';
      continue;
    }

    const leading = line.slice(0, firstNonSpace);
    const middle = line.slice(firstNonSpace, lastNonSpace + 1);
    const trailing = line.slice(lastNonSpace + 1);

    html += escapeHtml(leading);

    let middleHtml = '';
    let buffer = '';
    let currentColor = null;

    const flush = () => {
      if (!buffer) return;
      const safe = escapeHtml(buffer);
      middleHtml += currentColor
        ? `<span style="color:${currentColor};font-weight:700">${safe}</span>`
        : safe;
      buffer = '';
    };

    for (const char of middle) {
      let color = null;
      if (char === ' ') {
        color = null;
      } else if (inBasket) {
        color = basketColor;
      } else if (char === fillChar) {
        color = fillColor;
      } else {
        color = outlineColor;
      }
      if (color !== currentColor) {
        flush();
        currentColor = color;
      }
      buffer += char;
    }
    flush();

    html += `<span style="background-color:${skyColor}">${middleHtml}</span>`;
    html += escapeHtml(trailing);

    if (lineIndex < lines.length - 1) html += '\n';
  }

  return html;
}

export function placeBalloons(container, balloonSpecs, config = {}) {
  if (!container) return () => {};

  const resolved = { ...BALLOON_CONFIG, ...config };
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const balloons = [];
  container.innerHTML = '';

  for (const spec of balloonSpecs) {
    const shape = SHAPES[spec.shape] || SHAPES.small;
    const palette = BALLOON_PALETTES[spec.palette] || BALLOON_PALETTES.red;
    const fontSize = spec.fontSizePx ?? 12;
    const depth = clamp(spec.depth ?? 1, 0, 1);
    const depthTint = (1 - depth) * resolved.atmStrength;

    const el = document.createElement('div');
    el.className = 'ascii-balloon';
    el.style.left = spec.x;
    el.style.top = spec.y;
    el.style.fontSize = `${fontSize}px`;
    el.style.transform = 'translateX(-50%)';
    el.innerHTML = renderBalloonHTML(
      shape.art,
      shape.fillChar,
      shape.basketLines,
      palette,
      depthTint,
      resolved.skyColor,
    );
    container.appendChild(el);

    balloons.push({
      el,
      fontSize,
      bobPhase: Math.random() * Math.PI * 2,
      periodMs: resolved.bobPeriodMs * rand(
        1 - resolved.bobPeriodJitter,
        1 + resolved.bobPeriodJitter,
      ),
    });
  }

  if (reduceMotion) return () => {};

  let frameId = 0;
  const tick = (now) => {
    for (const balloon of balloons) {
      const omega = (2 * Math.PI) / balloon.periodMs;
      const amp = balloon.fontSize * resolved.bobFraction;
      const yOffset = Math.sin(now * omega + balloon.bobPhase) * amp;
      balloon.el.style.transform = `translate(-50%, ${yOffset}px)`;
    }
    frameId = requestAnimationFrame(tick);
  };

  frameId = requestAnimationFrame(tick);
  return () => {
    cancelAnimationFrame(frameId);
    for (const balloon of balloons) balloon.el.remove();
    balloons.length = 0;
  };
}
