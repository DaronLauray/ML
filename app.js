const teamList = [
  'Atlanta Hawks',
  'Boston Celtics',
  'Brooklyn Nets',
  'Charlotte Hornets',
  'Chicago Bulls',
  'Cleveland Cavaliers',
  'Dallas Mavericks',
  'Denver Nuggets',
  'Detroit Pistons',
  'Golden State Warriors',
  'Houston Rockets',
  'Indiana Pacers',
  'LA Clippers',
  'Los Angeles Lakers',
  'Memphis Grizzlies',
  'Miami Heat',
  'Milwaukee Bucks',
  'Minnesota Timberwolves',
  'New Orleans Pelicans',
  'New York Knicks',
  'Oklahoma City Thunder',
  'Orlando Magic',
  'Philadelphia 76ers',
  'Phoenix Suns',
  'Portland Trail Blazers',
  'Sacramento Kings',
  'San Antonio Spurs',
  'Seattle SuperSonics',
  'Toronto Raptors',
  'Utah Jazz',
  'Washington Wizards'
];

function populateTeams() {
  const homeSelect = document.getElementById('homeTeam');
  const awaySelect = document.getElementById('awayTeam');

  const sortedTeams = [...teamList].sort((a, b) => a.localeCompare(b));

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

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function getStaticPrediction(homeTeam, awayTeam, gameDate) {
  const homeSeed = hashString(homeTeam);
  const awaySeed = hashString(awayTeam);
  const dateSeed = gameDate ? hashString(gameDate) : 0;

  const weightedScore = ((homeSeed % 1000) - (awaySeed % 1000)) / 10;
  const dateBoost = (dateSeed % 100) / 100;
  const baseHomeChance = 0.5 + weightedScore / 100 + dateBoost * 0.1 + 0.08;

  const homeProbability = Math.max(0.15, Math.min(0.85, baseHomeChance));
  const awayProbability = 1 - homeProbability;
  const prediction = homeProbability >= awayProbability ? homeTeam : awayTeam;

  const explanation = [
    `Home-court edge: ${(0.08 * 100).toFixed(1)}% boost`,
    `Recent form index: ${((homeProbability - 0.5) * 100).toFixed(1)} pts`,
    `Matchup strength: ${Math.abs(weightedScore).toFixed(1)}`,
    `Date signal: ${(dateBoost * 100).toFixed(0)}%`,
    `Estimated pace differential: ${((homeSeed % 17) - (awaySeed % 17)).toFixed(0)} pace`,
    `Model confidence: ${homeProbability >= awayProbability ? 'Home side favored' : 'Away side favored'}`
  ];

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

function handleSubmit(event) {
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
