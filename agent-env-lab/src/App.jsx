import React, { useEffect, useMemo, useRef, useState } from 'react'

/* -----------------------------
 * Utils
 * ----------------------------- */
const clamp = (v,a,b)=>Math.max(a,Math.min(b,v))
const randInt = (a,b)=>a+Math.floor(Math.random()*(b-a+1))
const choice = (arr)=>arr[Math.floor(Math.random()*arr.length)]
const now = ()=>performance.now()
const manhattan = (a,b)=>Math.abs(a.x-b.x)+Math.abs(a.y-b.y)
const keyOf = (p)=>`${p.x},${p.y}`
const DIRS = [
  { name:'UP', dx:0, dy:-1 },
  { name:'RIGHT', dx:1, dy:0 },
  { name:'DOWN', dx:0, dy:1 },
  { name:'LEFT', dx:-1, dy:0 },
]

/* -----------------------------
 * Tuning knobs (feel free to tweak)
 * ----------------------------- */
const LOOP_PENALTY = 0.10;       // penalty for revisiting recent cells in rollouts
const LOOP_PENALTY_STEPS = 6;   // apply loop-penalty only in early rollout steps
const INFO_GAIN_WEIGHT = 0.02;  // reward per newly revealed cell in rollout
const NO_PROGRESS_TICKS = 18;   // if no distance improvement for this many ticks => stuck
const STUCK_MIN_HISTORY = 6;    // need at least this many steps in history to call it stuck

/* -----------------------------
 * Default config
 * ----------------------------- */
const DEFAULTS = {
  width: 15,
  height: 15,
  partialObs: true,
  obsRadius: 3,
  slipProb: 0.2,
  dynamicObstacles: false,
  episodic: true,
  granularity: 1,        // (placeholder; 1-cell moves)
  multiAgent: false,
  timeBudgetMs: 25,
  adversaryType: 'random', // 'random' | 'chaser'
  sensorNoise: 0.0,
}

/* -----------------------------
 * Env helpers
 * ----------------------------- */
function makeEnv(cfg){
  const width = cfg.width, height = cfg.height
  const obstacles = new Set()
  const obstacleCount = Math.floor(width*height*0.12)
  while (obstacles.size < obstacleCount) {
    const x = randInt(0,width-1), y = randInt(0,height-1)
    obstacles.add(`${x},${y}`)
  }
  const agent = { x:1, y:height-2 }
  const goal  = { x:width-2, y:1 }
  obstacles.delete(`${agent.x},${agent.y}`)
  obstacles.delete(`${goal.x},${goal.y}`)
  const adversary = cfg.multiAgent ? { x: width-2, y: height-2 } : null
  if (adversary) obstacles.delete(`${adversary.x},${adversary.y}`)

  const bestDist = manhattan(agent, goal)

  return {
    tick: 0,
    width, height,
    obstacles,
    agent, goal, adversary,
    reward: 0,
    done: false,
    seen: new Set([`${agent.x},${agent.y}`]),
    // NEW: loop/progress tracking
    recent: [keyOf(agent)],      // queue of recent positions
    bestDist,
    lastImprovedTick: 0,
  }
}

function cloneEnv(e){
  return {
    tick: e.tick,
    width: e.width, height: e.height,
    obstacles: new Set(e.obstacles),
    agent: { ...e.agent },
    goal: { ...e.goal },
    adversary: e.adversary ? { ...e.adversary } : null,
    reward: e.reward,
    done: e.done,
    seen: new Set(e.seen),
    recent: [...e.recent],
    bestDist: e.bestDist,
    lastImprovedTick: e.lastImprovedTick,
  }
}

function isBlocked(env,x,y){
  if (x<0||y<0||x>=env.width||y>=env.height) return true
  return env.obstacles.has(`${x},${y}`)
}

function stepEnvironment(env,cfg,action){
  if (env.done) return env
  const { agent, goal } = env

  // stochastic slip
  let a = action
  if (Math.random() < cfg.slipProb){
    const others = DIRS.filter(d=>d.name!==action.name)
    a = choice(others)
  }

  // (granularity placeholder: still 1-cell moves)
  const nx = agent.x + a.dx
  const ny = agent.y + a.dy
  if (!isBlocked(env,nx,ny)){
    agent.x = clamp(nx,0,env.width-1)
    agent.y = clamp(ny,0,env.height-1)
  }

  // reveal seen cells
  if (cfg.partialObs){
    const r = cfg.obsRadius
    for (let dy=-r; dy<=r; dy++){
      for (let dx=-r; dx<=r; dx++){
        const x = agent.x + dx, y = agent.y + dy
        if (x>=0&&y>=0&&x<env.width&&y<env.height){
          if (Math.random() < cfg.sensorNoise) continue
          env.seen.add(`${x},${y}`)
        }
      }
    }
  } else {
    if (env.seen.size < env.width*env.height){
      for (let y=0;y<env.height;y++)
        for (let x=0;x<env.width;x++) env.seen.add(`${x},${y}`)
    }
  }

  // dynamic obstacles
  if (cfg.dynamicObstacles && env.tick % 3 === 0){
    const moved = new Set()
    env.obstacles.forEach(key=>{
      const [ox,oy] = key.split(',').map(Number)
      const d = choice(DIRS)
      const tx = clamp(ox+d.dx,0,env.width-1)
      const ty = clamp(oy+d.dy,0,env.height-1)
      if (!isBlocked(env,tx,ty)
        && !(env.agent.x===tx && env.agent.y===ty)
        && !(env.goal.x===tx && env.goal.y===ty)
      ){
        moved.add(`${tx},${ty}`)
      } else {
        moved.add(`${ox},${oy}`)
      }
    })
    env.obstacles = moved
    env.obstacles.delete(`${env.agent.x},${env.agent.y}`)
    env.obstacles.delete(`${env.goal.x},${env.goal.y}`)
    if (env.adversary) env.obstacles.delete(`${env.adversary.x},${env.adversary.y}`)
  }

  // adversary
  let advPenalty = 0
  if (env.adversary){
    const adv = env.adversary
    let advMove
    if (cfg.adversaryType === 'chaser'){
      const candidates = DIRS
        .map(d=>({ d, x: clamp(adv.x+d.dx,0,env.width-1), y: clamp(adv.y+d.dy,0,env.height-1) }))
        .filter(p=>!isBlocked(env,p.x,p.y))
      const best = candidates.reduce((b,c)=>(
        manhattan(c,agent) < manhattan(b,agent) ? c : b
      ), candidates[0] || { x:adv.x, y:adv.y })
      advMove = best
    } else {
      const dirs = DIRS
        .map(d=>({ d, x: clamp(adv.x+d.dx,0,env.width-1), y: clamp(adv.y+d.dy,0,env.height-1) }))
        .filter(p=>!isBlocked(env,p.x,p.y))
      advMove = choice(dirs) || { x:adv.x, y:adv.y }
    }
    adv.x = advMove.x; adv.y = advMove.y
    if (adv.x===agent.x && adv.y===agent.y) advPenalty = -5
  }

  // rewards
  let r = -0.05
  if (agent.x===goal.x && agent.y===goal.y) r += 10
  r += advPenalty
  env.reward += r
  env.tick += 1

  // NEW: track progress to detect stagnation
  const dist = manhattan(agent, goal)
  if (dist < env.bestDist){
    env.bestDist = dist
    env.lastImprovedTick = env.tick
  }

  // push recent position (tabu memory)
  env.recent.push(keyOf(agent))
  if (env.recent.length > 16) env.recent.shift()

  if (cfg.episodic && agent.x===goal.x && agent.y===goal.y) env.done = true
  return env
}

/* -----------------------------
 * Stuck detection & info gain
 * ----------------------------- */
function isABAB(recent){
  if (recent.length < 4) return false
  const n = recent.length
  return recent[n-1]===recent[n-3] && recent[n-2]===recent[n-4]
}

function isStuck(env){
  if (env.recent.length < STUCK_MIN_HISTORY) return false
  // No progress for a while OR classic A-B-A-B loop
  const noProgress = (env.tick - env.lastImprovedTick) >= NO_PROGRESS_TICKS
  const abab = isABAB(env.recent)
  // Few unique cells in last 6 (pacing in a corner)
  const tail = env.recent.slice(-6)
  const uniq = new Set(tail)
  const tinyVar = uniq.size <= 2
  return noProgress || abab || tinyVar
}

function unseenAroundFrom(pos, env, cfg){
  // Count unseen cells in obsRadius around a hypothetical position
  const r = cfg.obsRadius
  let c = 0
  for (let dy=-r; dy<=r; dy++){
    for (let dx=-r; dx<=r; dx++){
      const x = pos.x + dx, y = pos.y + dy
      if (x>=0&&y>=0&&x<env.width&&y<env.height){
        if (!env.seen.has(`${x},${y}`)) c++
      }
    }
  }
  return c
}

/* -----------------------------
 * Anytime Monte-Carlo rollouts (with loop penalty + info gain)
 * ----------------------------- */
function evaluateAction(env,cfg,action,horizon=12){
  const sim = cloneEnv(env)
  let total = 0
  const startSeen = sim.seen.size
  const recentSet = new Set(sim.recent)  // for quick loop penalty checks

  for (let t=0; t<horizon && !sim.done; t++){
    const a = (t===0 ? action : choice(DIRS))
    stepEnvironment(sim,cfg,a)

    // Distance shaping toward goal
    total += -manhattan(sim.agent, sim.goal) * 0.02

    // Info gain: reward revealing new cells (VoI-lite)
    const gain = sim.seen.size - startSeen
    total += gain * INFO_GAIN_WEIGHT

    // Early loop-avoidance penalty if revisiting very recent cells
    if (t < LOOP_PENALTY_STEPS && recentSet.has(keyOf(sim.agent))){
      total -= LOOP_PENALTY
    }

    if (sim.agent.x===sim.goal.x && sim.agent.y===sim.goal.y){
      total += 5
    }
  }
  return total
}

function pickAction(env,cfg,budgetMs){
  const start = now()
  const scores = new Map(DIRS.map(d=>[d.name,{s:0,n:0}]))
  const horizon = cfg.partialObs ? 14 : 10
  let iters = 0

  // If stuck, try an information-seeking fallback first
  if (isStuck(env) && cfg.partialObs){
    // Choose the move that reveals the most unseen cells from next position
    let best = DIRS[0], bestInfo = -Infinity
    for (const d of DIRS){
      const nx = clamp(env.agent.x + d.dx, 0, env.width-1)
      const ny = clamp(env.agent.y + d.dy, 0, env.height-1)
      if (isBlocked(env, nx, ny)) continue
      const info = unseenAroundFrom({x:nx,y:ny}, env, cfg)
      if (info > bestInfo){ bestInfo = info; best = d }
    }
    // If all blocked (rare), fall through to MC search
    if (bestInfo > -Infinity){
      return { action: best, estimates: {}, iters: 0, mode: 'explore' }
    }
  }

  // Normal anytime rollouts
  while (now() - start < budgetMs){
    for (const d of DIRS){
      const val = evaluateAction(env,cfg,d,horizon)
      const rec = scores.get(d.name); rec.s += val; rec.n += 1
      iters++
      if (now() - start >= budgetMs) break
    }
  }
  let best = DIRS[0], bestMean = -Infinity
  for (const d of DIRS){
    const rec = scores.get(d.name)
    const mean = rec.n ? rec.s/rec.n : -Infinity
    if (mean > bestMean){ bestMean = mean; best = d }
  }

  // Tiny epsilon to avoid deterministic ties → reduces cycling
  if (Math.random() < 0.05){
    const sorted = [...DIRS].sort((a,b)=>{
      const ma = scores.get(a.name); const mb = scores.get(b.name)
      const aM = ma.n ? ma.s/ma.n : -Infinity
      const bM = mb.n ? mb.s/mb.n : -Infinity
      return bM - aM
    })
    best = choice(sorted.slice(0, Math.max(2, Math.floor(sorted.length/2))))
  }

  return {
    action: best,
    estimates: Object.fromEntries([...scores.entries()].map(([k,v])=>[k, v.n? v.s/v.n : 0])),
    iters,
    mode: 'mc',
  }
}

/* -----------------------------
 * React component
 * ----------------------------- */
export default function App(){
  const [cfg,setCfg] = useState(DEFAULTS)
  const [env,setEnv] = useState(()=>makeEnv(DEFAULTS))
  const [playing,setPlaying] = useState(false)
  const [lastDecision,setLastDecision] = useState(null)
  const canvasRef = useRef(null)
  const rafRef = useRef()

  useEffect(()=>{ setEnv(makeEnv(cfg)) }, [cfg.width,cfg.height,cfg.multiAgent])

  // draw
  useEffect(()=>{
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const cell = 28
    canvas.width = env.width * cell
    canvas.height = env.height * cell

    // bg
    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0,0,canvas.width,canvas.height)

    // grid & seen
    for (let y=0;y<env.height;y++){
      for (let x=0;x<env.width;x++){
        const seen = env.seen.has(`${x},${y}`) || !cfg.partialObs
        const px = x*cell, py = y*cell
        ctx.fillStyle = seen ? '#e2e8f0' : '#cbd5e1'
        ctx.fillRect(px,py,cell-1,cell-1)
      }
    }

    // obstacles
    env.obstacles.forEach(key=>{
      const [x,y] = key.split(',').map(Number)
      const seen = env.seen.has(key) || !cfg.partialObs
      if (!seen) return
      const px = x*cell, py = y*cell
      ctx.fillStyle = '#475569'
      ctx.fillRect(px+3,py+3,cell-7,cell-7)
    })

    // goal
    ctx.fillStyle = '#10b981'
    ctx.beginPath()
    ctx.arc(env.goal.x*cell+cell/2, env.goal.y*cell+cell/2, cell*0.28, 0, Math.PI*2)
    ctx.fill()

    // adversary
    if (env.adversary){
      ctx.fillStyle = '#ef4444'
      ctx.fillRect(env.adversary.x*cell+6, env.adversary.y*cell+6, cell-12, cell-12)
    }

    // agent
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.arc(env.agent.x*cell+cell/2, env.agent.y*cell+cell/2, cell*0.32, 0, Math.PI*2)
    ctx.fill()

    // obs radius
    if (cfg.partialObs){
      ctx.strokeStyle = 'rgba(59,130,246,0.45)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(env.agent.x*cell+cell/2, env.agent.y*cell+cell/2, (cfg.obsRadius+0.5)*cell, 0, Math.PI*2)
      ctx.stroke()
    }
  }, [env,cfg])

  const step = ()=>{
    if (env.done) return
    const decision = pickAction(env, cfg, cfg.timeBudgetMs)
    const next = cloneEnv(env)
    stepEnvironment(next, cfg, decision.action)
    setEnv(next)
    setLastDecision({ ...decision, tick: next.tick, reward: next.reward })
  }

  useEffect(()=>{
    if (!playing) return
    let last = 0
    const loop = (t)=>{
      if (t - last > 160){
        step(); last = t
        if (env.done){ setPlaying(false); return }
      }
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return ()=> cancelAnimationFrame(rafRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, env.done, cfg])

  const reset = ()=>{ setEnv(makeEnv(cfg)); setLastDecision(null) }
  const randomize = ()=>{ setEnv(makeEnv({ ...cfg, width: randInt(10,18), height: randInt(10,18) })) }

  const perfLabel = useMemo(()=>(
    env.done ? 'Episode complete'
      : cfg.episodic ? 'Achievement task: reach goal'
      : 'Maintenance task: minimize cost'
  ), [cfg.episodic, env.done])

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="title">Agent–Environment Lab</h1>
        <p className="help">Toggle properties to see how the agent adapts. New: loop-avoidance, info-seeking fallback, no-progress detector.</p>

        <div className="control-group">
          <label className="control-title">Observability</label>
          <div className="row">
            <input type="checkbox" checked={cfg.partialObs}
              onChange={e=>setCfg(v=>({ ...v, partialObs: e.target.checked }))}/>
            <span className="small">Partial (fog-of-war)</span>
          </div>
          {cfg.partialObs && (
            <div style={{marginTop:8}}>
              <div className="small">Radius: {cfg.obsRadius}</div>
              <input type="range" min="1" max="6" value={cfg.obsRadius}
                onChange={e=>setCfg(v=>({ ...v, obsRadius: +e.target.value }))}/>
              <div className="small" style={{marginTop:6}}>Sensor noise: {cfg.sensorNoise.toFixed(2)}</div>
              <input type="range" min="0" max="0.3" step="0.01" value={cfg.sensorNoise}
                onChange={e=>setCfg(v=>({ ...v, sensorNoise: +e.target.value }))}/>
            </div>
          )}
        </div>

        <div className="control-group">
          <label className="control-title">Determinism</label>
          <div className="small">Slip probability: {cfg.slipProb.toFixed(2)}</div>
          <input type="range" min="0" max="0.8" step="0.01" value={cfg.slipProb}
            onChange={e=>setCfg(v=>({ ...v, slipProb: +e.target.value }))}/>
        </div>

        <div className="control-group" style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
          <div>
            <label className="control-title">Dynamics</label>
            <div className="row">
              <input type="checkbox" checked={cfg.dynamicObstacles}
                onChange={e=>setCfg(v=>({ ...v, dynamicObstacles: e.target.checked }))}/>
              <span className="small">Dynamic obstacles</span>
            </div>
          </div>
          <div>
            <label className="control-title">Horizon</label>
            <div className="row">
              <input type="checkbox" checked={cfg.episodic}
                onChange={e=>setCfg(v=>({ ...v, episodic: e.target.checked }))}/>
              <span className="small">Episodic (end at goal)</span>
            </div>
          </div>
        </div>

        <div className="control-group" style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
          <div>
            <label className="control-title">Action space</label>
            <div className="small">Granularity: {cfg.granularity}×</div>
            <input type="range" min="1" max="3" step="1" value={cfg.granularity}
              onChange={e=>setCfg(v=>({ ...v, granularity: +e.target.value }))}/>
            <div className="small">Higher = finer sub-steps (continuous-ish)</div>
          </div>
          <div>
            <label className="control-title">Multi-agent</label>
            <div className="row">
              <input type="checkbox" checked={cfg.multiAgent}
                onChange={e=>setCfg(v=>({ ...v, multiAgent: e.target.checked }))}/>
              <span className="small">Enable adversary</span>
            </div>
            {cfg.multiAgent && (
              <select style={{marginTop:6, width:'100%'}} value={cfg.adversaryType}
                onChange={e=>setCfg(v=>({ ...v, adversaryType: e.target.value }))}>
                <option value="random">Random walker</option>
                <option value="chaser">Chaser (greedy)</option>
              </select>
            )}
          </div>
        </div>

        <div className="control-group">
          <label className="control-title">Real-time planning budget</label>
          <div className="small">{cfg.timeBudgetMs} ms / step</div>
          <input type="range" min="2" max="250" step="1" value={cfg.timeBudgetMs}
            onChange={e=>setCfg(v=>({ ...v, timeBudgetMs: +e.target.value }))}/>
        </div>

        <div className="buttons">
          <button className="btn btn-primary" onClick={step}>Step</button>
          <button className="btn btn-green" onClick={()=>setPlaying(p=>!p)}>{playing ? 'Pause' : 'Play'}</button>
          <button className="btn btn-dark" onClick={reset}>Reset</button>
          <button className="btn btn-outline" onClick={randomize}>Randomize</button>
        </div>

        <div className="section status">
          <div><strong>Status</strong></div>
          <div>Tick: {env.tick}</div>
          <div>Cumulative reward: {env.reward.toFixed(2)}</div>
          <div>{perfLabel}</div>
          {lastDecision && (
            <div>
              Last action: <code>{lastDecision.action.name}</code> · iters: {lastDecision.iters} · mode: {lastDecision.mode || 'mc'}<br/>
              {lastDecision.estimates && (
                <code>
                  U {lastDecision.estimates.UP?.toFixed(2)} ·
                  R {lastDecision.estimates.RIGHT?.toFixed(2)} ·
                  D {lastDecision.estimates.DOWN?.toFixed(2)} ·
                  L {lastDecision.estimates.LEFT?.toFixed(2)}
                </code>
              )}
            </div>
          )}
        </div>
      </aside>

      <main className="canvas-wrap">
        <div className="canvas-label">Environment</div>
        <canvas ref={canvasRef} />
      </main>
    </div>
  )
}
