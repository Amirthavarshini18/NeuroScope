const API_BASE = new URLSearchParams(window.location.search).get('api') || 'http://127.0.0.1:8002'
const elText = document.getElementById('text')
const btn = document.getElementById('analyze')
const status = document.getElementById('status')
const result = document.getElementById('result')
const emotionEl = document.getElementById('emotion')
const messageEl = document.getElementById('message')
const moodCanvas = document.getElementById('moodChart')
let moodChart = null

btn.addEventListener('click', async () => {
  const text = elText.value.trim()
  if(!text){
    status.textContent = 'Enter some text to analyze.'
    return
  }
  status.textContent = 'Analyzing...'
  try{
    const resp = await fetch(API_BASE + '/predict_text', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    })
    const j = await resp.json()
    emotionEl.textContent = j.emotion

    const rec = await fetch(API_BASE + '/recommend', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({emotion: j.emotion})
    }).then(r => r.json())

    messageEl.textContent = rec.recommendation.message
    result.classList.remove('hidden')

    // compute valence locally using same heuristic as backend
    function mapEmotionToValence(emotion){
      const m = (emotion||'').toLowerCase()
      const mapping = {
        'motivated': 0.8,
        'joy': 0.9,
        'neutral': 0.0,
        'low_mood': -0.6,
        'sadness': -0.6,
        'anxiety': -0.7,
        'fatigue': -0.4,
        'overwhelmed': -0.8,
        'anger': -0.7,
        'fear': -0.7
      }
      return mapping[m] !== undefined ? mapping[m] : 0.0
    }

    const valence = mapEmotionToValence(j.emotion)
    const productivity = Math.round(((valence + 1) * 5) * 10) / 10

    try{
      const logResp = await fetch(API_BASE + '/log', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text, productivity})
      })
      if(!logResp.ok){
        console.warn('/log returned', logResp.status)
      }
    }catch(err){
      console.warn('Failed to POST /log', err)
    }

    try{ await loadLogs() }catch(e){/* ignore chart errors */}

    status.textContent = 'Done.'
  }catch(e){
    status.textContent = `Error connecting to API — start backend at ${API_BASE}`
    console.error(e)
  }
})


async function loadLogs(){
  try{
    const resp = await fetch(API_BASE + '/logs')
    if(!resp.ok){
      console.warn('/logs returned', resp.status)
      return
    }
    const data = await resp.json()
    const labels = data.map(d => new Date(d.timestamp).toLocaleString())
    const productivity = data.map(d => (d.productivity === null || d.productivity === undefined) ? null : d.productivity)

    if(!moodChart){
      moodChart = new Chart(moodCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            {label: 'Productivity', data: productivity, borderColor: '#36a2eb', backgroundColor: 'rgba(54,162,235,0.1)', tension: 0.2}
          ]
        },
        options: {
          scales: {
            y: {type: 'linear', position: 'left', suggestedMin: 0, suggestedMax: 10, title: {display: true, text: 'Productivity'}}
          }
        }
      })
    }else{
      moodChart.data.labels = labels
      moodChart.data.datasets[0].data = productivity
      moodChart.update()
    }
  }catch(e){
    console.error('Failed to load logs for chart', e)
  }
}

loadLogs()
setInterval(loadLogs, 20000)
