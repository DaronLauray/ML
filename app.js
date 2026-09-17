let modelData = null;
let teamProfiles = null;

function normalizeTeamName(name) {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findProfileKey(teamName, profileMap) {
  const target = normalizeTeamName(teamName);
  if (!target) {
    return null;
  }

  const exact = Object.keys(profileMap).find((key) => normalizeTeamName(key) === target);
  if (exact) {
    return exact;
  }

  return Object.keys(profileMap).find((key) => {
    const keyName = normalizeTeamName(key);
    return keyName.includes(target) || target.includes(keyName);
  }) || null;
}

async function loadStaticModel() {
  if (modelData && teamProfiles) {
    return;
  }

  const [modelResponse, teamsResponse] = await Promise.all([
    fetch('model_export.json'),
    fetch('team_profiles.json')
  ]);

  if (!modelResponse.ok || !teamsResponse.ok) {
    throw new Error('The deployed Pages model files could not be loaded.');
  }

  modelData = await modelResponse.json();
  teamProfiles = await teamsResponse.json();
}

function evaluateTree(treeNodes, featureRow, nodeIndex = 0) {
  const node = treeNodes[nodeIndex];

  if (node.is_leaf) {
    const value = node.value || [0, 0];
    const total = value.reduce((sum, item) => sum + Number(item || 0), 0) || 1;
    return [
      Number(value[0] || 0) / total,
      Number(value[1] || 0) / total
    ];
  }

  const featureName = modelData.feature_names[node.feature_index];
  const value = Number(featureRow[featureName] ?? 0);
  const nextIndex = value <= node.threshold ? node.left : node.right;
  return evaluateTree(treeNodes, featureRow, nextIndex);
}

function getStaticPrediction(homeTeam, awayTeam) {
  if (!modelData || !teamProfiles) {
    throw new Error('The model has not finished loading.');
  }

  const homeProfileKey = findProfileKey(homeTeam, teamProfiles);
  const awayProfileKey = findProfileKey(awayTeam, teamProfiles);

  if (!homeProfileKey || !awayProfileKey) {
    throw new Error(`Could not find team data for ${homeTeam} or ${awayTeam}.`);
  }

  const featureRow = {};
  modelData.feature_names.forEach((featureName) => {
    const baseFeature = featureName.replace(/^diff_/, '');
    const homeValue = Number(teamProfiles[homeProfileKey][baseFeature] || 0);
    const awayValue = Number(teamProfiles[awayProfileKey][baseFeature] || 0);
    featureRow[featureName] = homeValue - awayValue;
  });

  let homeProbability = 0;
  let awayProbability = 0;

  modelData.trees.forEach((treeNodes) => {
    const probabilities = evaluateTree(treeNodes, featureRow, 0);
    homeProbability += probabilities[1];
    awayProbability += probabilities[0];
  });

  homeProbability /= modelData.trees.length || 1;
  awayProbability /= modelData.trees.length || 1;

  const prediction = homeProbability >= awayProbability ? homeTeam : awayTeam;

  const featureDiffs = [...modelData.feature_names]
    .map((featureName) => ({
      name: featureName.replace(/^diff_/, ''),
      value: Math.abs(Number(featureRow[featureName] || 0))
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);

  const explanation = featureDiffs.map((item) => `${item.name}: ${Number(featureRow[`diff_${item.name}`] || 0).toFixed(2)}`);

  return {
    prediction,
    home_probability: homeProbability,
    away_probability: awayProbability,
    explanation,
    home_team: homeTeam,
    away_team: awayTeam
  };
}

function showError(message) {
  const errorBox = document.getElementById('errorBox');
  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

function hideError() {
  document.getElementById('errorBox').style.display = 'none';
}

function showLoading(isLoading) {
  const loading = document.getElementById('loading');
  loading.classList.toggle('show', isLoading);
}

function setResult(data) {
  const result = document.getElementById('result');
  result.classList.add('show');

  const homeName = data.home_team || 'Home';
  const awayName = data.away_team || 'Away';
  document.getElementById('predictionText').textContent = `${data.prediction} wins`;
  document.getElementById('homeProbLabel').textContent = `${homeName} Win`;
  document.getElementById('awayProbLabel').textContent = `${awayName} Win`;
  document.getElementById('homeProbability').textContent = `${(data.home_probability * 100).toFixed(1)}%`;
  document.getElementById('awayProbability').textContent = `${(data.away_probability * 100).toFixed(1)}%`;

  const explanationList = document.getElementById('explanationList');
  explanationList.innerHTML = '';
  (data.explanation || []).slice(0, 6).forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    explanationList.appendChild(li);
  });
}

async function populateTeams() {
  await loadStaticModel();

  const homeSelect = document.getElementById('homeTeam');
  const awaySelect = document.getElementById('awayTeam');
  const sortedTeams = Object.keys(teamProfiles).sort((a, b) => a.localeCompare(b));

  sortedTeams.forEach((team) => {
    const homeOption = document.createElement('option');
    homeOption.value = team;
    homeOption.textContent = team;
    homeSelect.appendChild(homeOption);

    const awayOption = document.createElement('option');
    awayOption.value = team;
    awayOption.textContent = team;
    awaySelect.appendChild(awayOption);
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  hideError();

  const homeTeam = document.getElementById('homeTeam').value;
  const awayTeam = document.getElementById('awayTeam').value;
  const gameDate = document.getElementById('gameDate').value;

  if (!homeTeam || !awayTeam) {
    showError('Please select both home and away teams.');
    return;
  }

  if (homeTeam === awayTeam) {
    showError('Home and away teams must be different.');
    return;
  }

  showLoading(true);

  try {
    await loadStaticModel();
    const data = getStaticPrediction(homeTeam, awayTeam, gameDate);
    setResult(data);
  } catch (error) {
    showError(error.message || 'Unable to get a prediction right now.');
  } finally {
    showLoading(false);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  populateTeams();
  const form = document.getElementById('predictionForm');
  form.addEventListener('submit', handleSubmit);
});
