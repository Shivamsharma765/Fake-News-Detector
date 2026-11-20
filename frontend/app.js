// FactLens - Client Logic & Interactive Fact-Checking
function initFactLens() {
  // DOM Elements
  const newsInput = document.getElementById('newsInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  if (!newsInput || !analyzeBtn) return;

  const btnText = analyzeBtn.querySelector('.btn-text');
  const spinner = analyzeBtn.querySelector('.spinner');

  const modeLive = document.getElementById('modeLive');
  const modeNlp = document.getElementById('modeNlp');

  const showcaseView = document.getElementById('showcaseView');
  const resultsView = document.getElementById('resultsView');
  const aiCoreOrb = document.getElementById('aiCoreOrb');
  const btnBackOverview = document.getElementById('btnBackOverview');

  const verdictDisplayCard = document.getElementById('verdictDisplayCard');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictLabel = document.getElementById('verdictLabel');
  const verdictHeadline = document.getElementById('verdictHeadline');
  const verdictExplanation = document.getElementById('verdictExplanation');

  const debunkCallout = document.getElementById('debunkCallout');
  const debunkSourceText = document.getElementById('debunkSourceText');

  const gaugeCircle = document.getElementById('gaugeCircle');
  const gaugePct = document.getElementById('gaugePct');
  const valReal = document.getElementById('valReal');
  const valFake = document.getElementById('valFake');
  const barReal = document.getElementById('barReal');
  const barFake = document.getElementById('barFake');

  const liveArticlesPanel = document.getElementById('liveArticlesPanel');
  const articlesCardsGrid = document.getElementById('articlesCardsGrid');
  const articlesFoundCount = document.getElementById('articlesFoundCount');

  const nlpVocabPanel = document.getElementById('nlpVocabPanel');
  const vocabCoveragePct = document.getElementById('vocabCoveragePct');
  const vocabPillsBox = document.getElementById('vocabPillsBox');

  // Modals & Navigation
  const navHow = document.getElementById('navHow');
  const modalHow = document.getElementById('modalHow');
  const closeHowBtn = document.getElementById('closeHowBtn');

  const navAbout = document.getElementById('navAbout');
  const modalAbout = document.getElementById('modalAbout');
  const closeAboutBtn = document.getElementById('closeAboutBtn');

  const tryNowBtn = document.getElementById('tryNowBtn');

  // State
  let currentMode = 'live'; // 'live' or 'nlp'

  // Engine Mode Switcher
  if (modeLive && modeNlp) {
    modeLive.addEventListener('click', (e) => {
      e.preventDefault();
      currentMode = 'live';
      modeLive.classList.add('active');
      modeNlp.classList.remove('active');
      newsInput.placeholder = 'Paste a news link, claim, or text...';
    });

    modeNlp.addEventListener('click', (e) => {
      e.preventDefault();
      currentMode = 'nlp';
      modeNlp.classList.add('active');
      modeLive.classList.remove('active');
      newsInput.placeholder = 'Paste headline or article to test against 44k NLP benchmark...';
    });
  }

  // Example Chips
  document.querySelectorAll('.cat-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      e.preventDefault();
      const text = pill.getAttribute('data-text');
      if (text) {
        newsInput.value = text;
        analyzeNews();
      }
    });
  });

  // Try Now button
  if (tryNowBtn) {
    tryNowBtn.addEventListener('click', (e) => {
      e.preventDefault();
      newsInput.focus();
      newsInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  }

  // Enter Key on input
  newsInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      analyzeNews();
    }
  });

  // Analyze Button Click
  analyzeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    analyzeNews();
  });

  // Close & Reset Button
  if (btnBackOverview) {
    btnBackOverview.addEventListener('click', (e) => {
      e.preventDefault();
      if (resultsView) resultsView.classList.add('hidden');
      if (showcaseView) showcaseView.classList.remove('hidden');
      newsInput.focus();
    });
  }

  // Modal Handlers
  if (navHow && modalHow && closeHowBtn) {
    navHow.addEventListener('click', (e) => {
      e.preventDefault();
      modalHow.classList.remove('hidden');
    });
    closeHowBtn.addEventListener('click', () => modalHow.classList.add('hidden'));
  }

  if (navAbout && modalAbout && closeAboutBtn) {
    navAbout.addEventListener('click', (e) => {
      e.preventDefault();
      modalAbout.classList.remove('hidden');
    });
    closeAboutBtn.addEventListener('click', () => modalAbout.classList.add('hidden'));
  }

  // Close modal when clicking backdrop
  [modalHow, modalAbout].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.add('hidden');
        }
      });
    }
  });

  // Toast Notification helper
  function showToast(msg, isError = true) {
    let t = document.getElementById('factToast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'factToast';
      t.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:' + (isError ? '#DC2626' : '#059669') + ';color:#FFF;padding:12px 26px;border-radius:9999px;font-weight:600;font-size:0.92rem;box-shadow:0 10px 30px rgba(0,0,0,0.3);z-index:99999;transition:opacity 0.3s ease;pointer-events:none;';
      document.body.appendChild(t);
    }
    t.style.background = isError ? '#DC2626' : '#059669';
    t.textContent = msg;
    t.style.opacity = '1';
    setTimeout(() => { t.style.opacity = '0'; }, 4500);
  }

  // Main Analysis Request
  async function analyzeNews() {
    const text = newsInput.value.trim();
    if (!text) {
      newsInput.focus();
      return;
    }

    // Set Loading Animation
    if (btnText) btnText.textContent = 'Analyzing...';
    if (spinner) spinner.classList.remove('hidden');
    analyzeBtn.disabled = true;
    if (aiCoreOrb) aiCoreOrb.classList.add('analyzing');

    try {
      // Production-ready API routing:
      // Use relative endpoint '/api/predict' in production or when served from backend.
      // Use localhost:5000 only when embedded inside local Streamlit (port 8501/8502).
      let apiUrl = '/api/predict';
      const isLocal = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
      if (isLocal && window.location.port && window.location.port !== '5000') {
        apiUrl = 'http://127.0.0.1:5000/api/predict';
      }

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          mode: currentMode,
          model: 'Multinomial Naive Bayes'
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      console.error('FactLens analysis error:', err);
      showToast('⚠️ Could not connect to verification engine. Please ensure server.py is running on port 5000.');
    } finally {
      if (btnText) btnText.textContent = 'Analyze →';
      if (spinner) spinner.classList.add('hidden');
      analyzeBtn.disabled = false;
      if (aiCoreOrb) aiCoreOrb.classList.remove('analyzing');
    }
  }

  // Display Verification Report
  function displayResults(data) {
    if (!resultsView || !showcaseView) return;

    // Switch views
    showcaseView.classList.add('hidden');
    resultsView.classList.remove('hidden');

    const {
      verdict = 'UNVERIFIED',
      confidence = 0.5,
      headline = '',
      note = '',
      prob_real = 0.5,
      prob_fake = 0.5,
      articles = [],
      debunked_by = '',
      matched_words = [],
      unmatched_words = [],
      total_words = 0
    } = data;

    // Reset themes
    if (verdictDisplayCard) {
      verdictDisplayCard.className = 'verdict-display-card';

      const isReal = verdict.includes('REAL');
      const isFake = verdict.includes('FAKE') || verdict.includes('HOAX');

      if (isReal) {
        verdictDisplayCard.classList.add('real-theme');
        if (verdictIcon) verdictIcon.textContent = '✓';
        if (verdictLabel) verdictLabel.textContent = verdict;
        if (gaugeCircle) gaugeCircle.style.stroke = 'var(--real-green)';
      } else if (isFake) {
        verdictDisplayCard.classList.add('fake-theme');
        if (verdictIcon) verdictIcon.textContent = '✕';
        if (verdictLabel) verdictLabel.textContent = verdict;
        if (gaugeCircle) gaugeCircle.style.stroke = 'var(--fake-red)';
      } else {
        verdictDisplayCard.classList.add('warn-theme');
        if (verdictIcon) verdictIcon.textContent = '⚠️';
        if (verdictLabel) verdictLabel.textContent = verdict;
        if (gaugeCircle) gaugeCircle.style.stroke = 'var(--warn-amber)';
      }
    }

    if (verdictHeadline) verdictHeadline.textContent = headline || 'Verification Findings';
    if (verdictExplanation) verdictExplanation.textContent = note || '';

    // Debunk Callout
    if (debunkCallout && debunkSourceText) {
      if (debunked_by) {
        debunkCallout.classList.remove('hidden');
        debunkSourceText.textContent = `Investigated and debunked by recognized fact-checkers & journalists (${debunked_by}).`;
      } else {
        debunkCallout.classList.add('hidden');
      }
    }

    // Gauge calculation
    const confVal = Math.round(confidence * 100);
    if (gaugePct) gaugePct.textContent = `${confVal}%`;

    const circumference = 2 * Math.PI * 40; // r=40
    const offset = circumference - (confVal / 100) * circumference;
    if (gaugeCircle) {
      gaugeCircle.style.strokeDashoffset = offset;
    }

    // Probability Bars
    const realPct = Math.round(prob_real * 100);
    const fakePct = Math.round(prob_fake * 100);
    if (valReal) valReal.textContent = `${realPct}%`;
    if (valFake) valFake.textContent = `${fakePct}%`;
    if (barReal) barReal.style.width = `${realPct}%`;
    if (barFake) barFake.style.width = `${fakePct}%`;

    // Mode Panels
    if (currentMode === 'live') {
      if (liveArticlesPanel) liveArticlesPanel.classList.remove('hidden');
      if (nlpVocabPanel) nlpVocabPanel.classList.add('hidden');

      if (articlesCardsGrid) {
        articlesCardsGrid.innerHTML = '';
        if (articlesFoundCount) articlesFoundCount.textContent = `${articles.length} verified reports found`;

        if (articles.length === 0) {
          articlesCardsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #64748B; padding: 2rem;">No matching articles found in global news indexes. Lack of reporting on sensational claims strongly signals fabricated rumors.</div>';
        } else {
          articles.slice(0, 6).forEach(art => {
            const card = document.createElement('div');
            card.className = 'article-card';
            card.innerHTML = `
              <div class="art-source-row">
                <span class="art-source-badge">${escapeHtml(art.source)}</span>
                <span class="art-date">${escapeHtml(art.date)}</span>
              </div>
              <div class="art-title">${escapeHtml(art.title)}</div>
              <a href="${escapeHtml(art.link)}" target="_blank" rel="noopener noreferrer" class="art-link">
                Read Verified Source Article &rarr;
              </a>
            `;
            articlesCardsGrid.appendChild(card);
          });
        }
      }
    } else {
      if (liveArticlesPanel) liveArticlesPanel.classList.add('hidden');
      if (nlpVocabPanel) nlpVocabPanel.classList.remove('hidden');

      if (vocabCoveragePct) {
        const coverage = total_words > 0 ? Math.round((matched_words.length / total_words) * 100) : 0;
        vocabCoveragePct.textContent = `${coverage}% Vocabulary Matched`;
      }

      if (vocabPillsBox) {
        vocabPillsBox.innerHTML = '';
        if (matched_words && matched_words.length > 0) {
          matched_words.forEach(w => {
            const pill = document.createElement('span');
            pill.className = 'nlp-pill match';
            pill.textContent = `✓ ${w}`;
            vocabPillsBox.appendChild(pill);
          });
        }
        if (unmatched_words && unmatched_words.length > 0) {
          unmatched_words.slice(0, 15).forEach(w => {
            const pill = document.createElement('span');
            pill.className = 'nlp-pill unmatch';
            pill.textContent = `? ${w}`;
            vocabPillsBox.appendChild(pill);
          });
        }
      }
    }

    // Scroll smoothly to results card
    resultsView.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function(m) {
      return {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      }[m];
    });
  }
}

// Ensure execution whether DOM is already loaded or still loading
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initFactLens);
} else {
  initFactLens();
}
