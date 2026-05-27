import { placeBalloons } from './balloons.js?v=2';
import { startBirds } from './birds.js?v=2';

// Each page declares its own balloons via window.SKY_BALLOONS before this
// module loads. Pages that omit it get no balloons.
const BALLOONS = Array.isArray(window.SKY_BALLOONS) ? window.SKY_BALLOONS : [];

const TEXT_SELECTORS = [
  '.profile-section h1',
  '.profile-section p',
  '.contact-section h2',
  '.contact-section p',
  '.site-footer',
  '.section-title',
  '.under-construction',
  '.work-item',
  '.project-item',
  '.project-card',
];

const ZONE_PADDING = 24; // px buffer around text bodies
const BIRDS_PER_VIEWPORT = 3;

function createLayer(id) {
  const existing = document.getElementById(id);
  if (existing) return existing;
  const layer = document.createElement('div');
  layer.id = id;
  layer.setAttribute('aria-hidden', 'true');
  document.body.prepend(layer);
  return layer;
}

function getDocumentHeight() {
  return Math.max(
    document.body.scrollHeight,
    document.body.offsetHeight,
    document.documentElement.scrollHeight,
    document.documentElement.offsetHeight,
  );
}

function computeTextZones() {
  const zones = [];
  for (const sel of TEXT_SELECTORS) {
    for (const el of document.querySelectorAll(sel)) {
      const rect = el.getBoundingClientRect();
      if (rect.height <= 0) continue;
      const top = rect.top + window.scrollY - ZONE_PADDING;
      const bottom = rect.bottom + window.scrollY + ZONE_PADDING;
      zones.push({ top, bottom });
    }
  }
  zones.sort((a, b) => a.top - b.top);
  const merged = [];
  for (const z of zones) {
    const last = merged[merged.length - 1];
    if (last && z.top <= last.bottom) {
      last.bottom = Math.max(last.bottom, z.bottom);
    } else {
      merged.push({ ...z });
    }
  }
  return merged;
}

function startSkyScene() {
  const balloonContainer = createLayer('balloon-container');
  const birdContainer = createLayer('bird-container');

  placeBalloons(balloonContainer, BALLOONS);

  const sizeContainers = () => {
    const h = `${getDocumentHeight()}px`;
    birdContainer.style.height = h;
    balloonContainer.style.height = h;
  };
  sizeContainers();

  const headerEl = document.querySelector('header');
  const headerHeight = headerEl ? headerEl.offsetHeight : 60;

  startBirds(birdContainer, {
    getAvoidZones: computeTextZones,
    getMaxBirds: () => {
      const ratio = getDocumentHeight() / window.innerHeight;
      return Math.max(3, Math.round(ratio * BIRDS_PER_VIEWPORT));
    },
    topMargin: headerHeight + 16,
  });

  // Resize-aware: keep both containers the height of the document, and let
  // computeTextZones re-read fresh DOM coords on each bird spawn.
  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(sizeContainers);
    ro.observe(document.body);
  }
  window.addEventListener('resize', sizeContainers);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startSkyScene, { once: true });
} else {
  startSkyScene();
}
