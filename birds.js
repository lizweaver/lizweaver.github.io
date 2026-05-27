export const BIRD_CONFIG = {
  maxBirds: 5,
  spawnGapMs: [1200, 4500],
  flapMs: [400, 500],
  baseSpeed: 55,
  speedJitter: 0.3,
  birdSizePx: 13,
  avoidZones: [],          // array of {top, bottom} in container/document coords
  getAvoidZones: null,     // optional function returning latest zones
  getMaxBirds: null,       // optional function returning latest maxBirds
  pctThroughZone: 0,       // never spawn across a text zone
  edgeMargin: 12,          // px buffer from container edges (used for bottom)
  topMargin: null,         // px from doc top below which birds may spawn (defaults to edgeMargin)
  zonePadding: 8,          // additional px around zones to leave clear
  eyes: ['@', 'O', '0', 'o'],
  bodies: ['::::', '####', '$$$$'],
  colors: ['#705D56', '#022B3A'],
};

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function pick(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function buildFrames(direction, eye, body) {
  if (direction === 'left') {
    return [
      ['  |\\/)', '  |/ )', `<${eye}${body}<`, '', ''].join('\n'),
      ['', '', `<${eye}${body}<`, '  | /', '  |/'].join('\n'),
    ];
  }

  return [
    [' (\\/|', ' ( \\|', `>${body}${eye}>`, '', ''].join('\n'),
    ['', '', `>${body}${eye}>`, '  \\ |', '   \\|'].join('\n'),
  ];
}

function resolveConfig(config) {
  return { ...BIRD_CONFIG, ...config };
}

export function startBirds(container, config = {}) {
  if (!container || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return () => {};
  }

  const resolved = resolveConfig(config);
  const birds = [];
  let frameId = 0;
  let spawnTimerId = 0;
  let last = performance.now();

  function viewportWidth() {
    return container.clientWidth || window.innerWidth;
  }

  function containerHeight() {
    return container.clientHeight || window.innerHeight;
  }

  function currentMaxBirds() {
    return typeof resolved.getMaxBirds === 'function'
      ? resolved.getMaxBirds()
      : resolved.maxBirds;
  }

  function currentAvoidZones() {
    const zones = typeof resolved.getAvoidZones === 'function'
      ? resolved.getAvoidZones()
      : resolved.avoidZones;
    if (!zones || zones.length === 0) return [];
    return [...zones].sort((a, b) => a.top - b.top);
  }

  function pickDirection() {
    if (birds.length === 0) return Math.random() < 0.5 ? 'left' : 'right';
    let left = 0;
    let right = 0;
    for (const bird of birds) {
      if (bird.direction === 'left') left += 1;
      else right += 1;
    }
    if (left === right) return Math.random() < 0.5 ? 'left' : 'right';
    const underrepresented = left < right ? 'left' : 'right';
    if (Math.random() < 0.75) return underrepresented;
    return underrepresented === 'left' ? 'right' : 'left';
  }

  function pickBaseY() {
    const h = containerHeight();
    const birdHeight = resolved.birdSizePx * 5;
    const margin = resolved.edgeMargin;
    const topMargin = resolved.topMargin ?? margin;
    const pad = resolved.zonePadding;
    const minY = topMargin;
    const maxY = Math.max(minY + 1, h - birdHeight - margin);

    const zones = currentAvoidZones();
    if (zones.length === 0) return rand(minY, maxY);

    if (Math.random() * 100 < resolved.pctThroughZone) {
      return rand(minY, maxY);
    }

    const gaps = [];
    const aboveEnd = zones[0].top - pad - birdHeight;
    if (aboveEnd >= minY) gaps.push({ start: minY, end: aboveEnd });

    for (let i = 0; i < zones.length - 1; i += 1) {
      const gapStart = zones[i].bottom + pad;
      const gapEnd = zones[i + 1].top - pad - birdHeight;
      if (gapEnd > gapStart) gaps.push({ start: gapStart, end: gapEnd });
    }

    const belowStart = zones[zones.length - 1].bottom + pad;
    if (belowStart <= maxY) gaps.push({ start: belowStart, end: maxY });

    if (gaps.length === 0) return rand(minY, maxY);

    const counts = gaps.map((gap) => {
      let count = 0;
      for (const bird of birds) {
        if (bird.baseY >= gap.start && bird.baseY <= gap.end) count += 1;
      }
      return count;
    });

    const minCount = Math.min(...counts);
    const candidates = gaps.filter((_, i) => counts[i] === minCount);
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    const size = Math.max(0, chosen.end - chosen.start);
    return size <= 0 ? rand(minY, maxY) : chosen.start + Math.random() * size;
  }

  function spawnBird() {
    const w = viewportWidth();
    const direction = pickDirection();
    const eye = pick(resolved.eyes);
    const body = pick(resolved.bodies);
    const frames = buildFrames(direction, eye, body);
    const baseY = pickBaseY();
    const el = document.createElement('div');

    el.className = 'ascii-bird';
    el.style.fontSize = `${resolved.birdSizePx}px`;
    el.style.color = pick(resolved.colors);
    el.textContent = frames[0];
    container.appendChild(el);

    const speed = resolved.baseSpeed
      * rand(1 - resolved.speedJitter, 1 + resolved.speedJitter)
      * (direction === 'left' ? -1 : 1);
    const startX = direction === 'left' ? w + 40 : -120;
    const flapMs = rand(resolved.flapMs[0], resolved.flapMs[1]);

    birds.push({
      el,
      frames,
      frame: 0,
      lastFlap: performance.now() + rand(0, flapMs),
      flapMs,
      x: startX,
      y: baseY,
      baseY,
      vx: speed,
      bobAmp: rand(3, 9),
      bobFreq: rand(0.001, 0.003),
      bobPhase: rand(0, Math.PI * 2),
      driftY: 0,
      driftVel: 0,
      direction,
    });
  }

  function scheduleNextSpawn() {
    const [minGap, maxGap] = resolved.spawnGapMs;
    spawnTimerId = window.setTimeout(() => {
      if (birds.length < currentMaxBirds()) spawnBird();
      scheduleNextSpawn();
    }, rand(minGap, maxGap));
  }

  function tick(now) {
    const dt = Math.min(50, now - last);
    last = now;
    const w = viewportWidth();

    for (let i = birds.length - 1; i >= 0; i -= 1) {
      const bird = birds[i];
      bird.x += (bird.vx * dt) / 1000;
      bird.driftVel += (Math.random() - 0.5) * 0.04;
      bird.driftVel *= 0.95;
      bird.driftY += bird.driftVel;
      bird.driftY = Math.max(-12, Math.min(12, bird.driftY));

      const bob = Math.sin(now * bird.bobFreq + bird.bobPhase) * bird.bobAmp;
      bird.y = bird.baseY + bob + bird.driftY;

      if (now - bird.lastFlap >= bird.flapMs) {
        bird.frame = (bird.frame + 1) % bird.frames.length;
        bird.el.textContent = bird.frames[bird.frame];
        bird.lastFlap = now;
      }

      bird.el.style.transform = `translate(${bird.x}px, ${bird.y}px)`;

      const isOffscreen = bird.vx < 0 ? bird.x < -160 : bird.x > w + 40;
      if (isOffscreen) {
        bird.el.remove();
        birds.splice(i, 1);
      }
    }

    frameId = requestAnimationFrame(tick);
  }

  scheduleNextSpawn();
  frameId = requestAnimationFrame(tick);

  return () => {
    window.clearTimeout(spawnTimerId);
    cancelAnimationFrame(frameId);
    for (const bird of birds) bird.el.remove();
    birds.length = 0;
  };
}
