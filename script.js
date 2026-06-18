const counters = document.querySelectorAll('.counter');

const animateCounter = (counter) => {
  const target = Number(counter.dataset.target);
  const displayed = counter.dataset.display || null;
  const duration = 1400;
  const start = performance.now();

  if (displayed) {
    counter.textContent = displayed;
    return;
  }

  const update = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.floor(eased * target);
    counter.textContent = `${value}+`;

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      counter.textContent = `${target}+`;
    }
  };

  requestAnimationFrame(update);
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const el = entry.target;
      if (el.classList.contains('counter') && !el.dataset.animated) {
        animateCounter(el);
        el.dataset.animated = 'true';
      }
    }
  });
}, { threshold: 0.4 });

counters.forEach((counter) => observer.observe(counter));

const stateEl = document.getElementById('live-state');
const detailEl = document.getElementById('live-detail');
const alertBadge = document.getElementById('alert-badge');
const alertText = document.getElementById('hero-alert-text');
const connectionPill = document.getElementById('connection-pill');
const closestObjectEl = document.getElementById('closest-object');
const closestDirectionEl = document.getElementById('closest-direction');
const freeDirectionEl = document.getElementById('free-direction');
const zoneCountEl = document.getElementById('zone-count');
const zoneListEl = document.getElementById('zone-list');

const zoneNames = ['Far Left', 'Left', 'Center', 'Right', 'Far Right'];
const liveStream = document.getElementById('live-stream');
const API_BASE = window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : '';

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function updateBadgeClass(rawState) {
  if (!rawState) {
    alertBadge.className = 'pill active';
    alertBadge.textContent = 'IDLE';
    return;
  }

  if (rawState.startsWith('DANGER')) {
    alertBadge.className = 'pill danger';
    alertBadge.textContent = 'DANGER';
  } else if (rawState.startsWith('BLOCKED')) {
    alertBadge.className = 'pill danger';
    alertBadge.textContent = 'BLOCKED';
  } else if (rawState.startsWith('CAUTION')) {
    alertBadge.className = 'pill warning';
    alertBadge.textContent = 'CAUTION';
  } else {
    alertBadge.className = 'pill active';
    alertBadge.textContent = 'ACTIVE';
  }
}

function renderZones(zones) {
  zoneCountEl.textContent = `${zones.length} zones`;
  zoneListEl.innerHTML = '';

  zones.forEach((zone, index) => {
    const row = document.createElement('div');
    row.className = 'zone-row';
    row.innerHTML = `
      <strong>${zoneNames[index]}</strong>
      <span>${zone.blocked ? 'Blocked' : zone.threat > 0.6 ? 'High risk' : zone.threat > 0.3 ? 'Watch' : 'Clear'}</span>
    `;
    zoneListEl.appendChild(row);
  });
}

function refreshStream() {
  if (!liveStream) return;
  liveStream.src = `${apiUrl('/api/frame')}?t=${Date.now()}`;
}

async function fetchStatus() {
  try {
    const response = await fetch(apiUrl('/api/status'));
    if (!response.ok) {
      throw new Error('Status unavailable');
    }

    const data = await response.json();
    stateEl.textContent = data.state || 'Monitoring';
    detailEl.textContent = data.detail || 'The detector is running normally.';
    alertText.textContent = data.state || 'Monitoring';
    closestObjectEl.textContent = data.closest_object || '—';
    closestDirectionEl.textContent = data.closest_direction || '—';
    freeDirectionEl.textContent = data.free_direction || '—';
    connectionPill.textContent = data.stream_connected ? '● Connected' : '● Waiting';
    connectionPill.style.color = data.stream_connected ? '#9bffcc' : '#ffbe3d';
    updateBadgeClass(data.state);
    renderZones(data.zones || []);
  } catch (error) {
    stateEl.textContent = 'Waiting for stream';
    detailEl.textContent = 'The detector will show the latest obstacle guidance here.';
    alertText.textContent = 'Monitoring';
    connectionPill.textContent = '● Waiting';
    connectionPill.style.color = '#ffbe3d';
    alertBadge.className = 'pill';
    alertBadge.textContent = 'OFFLINE';
  }
}

refreshStream();
fetchStatus();
setInterval(refreshStream, 800);
setInterval(fetchStatus, 1200);
