import { useState, useEffect, useCallback, useRef, useMemo } from "react";

/* ─── SKWID BRAND TOKENS ─────────────────────────────────────────────────── */
const B = {
  violet:    "#240854", violetMid: "#3A0E8A", violetDim: "#1A0640",
  black:     "#000000", white:     "#FFFFFF", offWhite:  "#F5F3FF",
  muted:     "#8B7BB0", faint:     "#2E1A5A", border:    "#2D1266",
  borderHi:  "#4A1EA0", green:     "#3DD68C", gold:      "#C8A96E",
  red:       "#E05252", blue:      "#6C8FE8",
  mono: "'IBM Plex Mono','Courier New',monospace",
  sans: "'Inter',system-ui,sans-serif",
};

/* ─── SKWID MARK ─────────────────────────────────────────────────────────── */
const SkwidMark = ({ size=28 }) => (
  <svg width={size} height={size*.85} viewBox="0 0 100 85" fill="none">
    <polygon points="50,2 72,15 72,40 50,53 28,40 28,15" fill={B.white}/>
    <polygon points="28,42 10,52 16,62 34,52" fill={B.white}/>
    <polygon points="16,62 2,72 10,82 24,72" fill={B.white}/>
    <polygon points="34,52 22,62 28,72 40,62" fill={B.white}/>
    <polygon points="72,42 90,52 84,62 66,52" fill={B.white}/>
    <polygon points="84,62 98,72 90,82 76,72" fill={B.white}/>
    <polygon points="66,52 78,62 72,72 60,62" fill={B.white}/>
    <polygon points="40,53 50,58 60,53 50,48" fill={B.white}/>
  </svg>
);

/* ─── FORMAT ─────────────────────────────────────────────────────────────── */
const fd  = n=>`$${Math.abs(n)>=1000?(n/1000).toFixed(1)+"k":Number(n).toFixed(0)}`;
const fdF = n=>`$${Number(n).toLocaleString("en-US",{minimumFractionDigits:0})}`;
const fp  = (n,d=1)=>`${(n*100).toFixed(d)}%`;
const fn  = (n,d=0)=>Number(n).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});

/* ─── RNG ────────────────────────────────────────────────────────────────── */
function bm(){ let u,v; do{u=Math.random();}while(!u); do{v=Math.random();}while(!v); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function rNorm(m=0,s=1){ return m+s*bm(); }
function rGamma(sh){ if(sh<1) return rGamma(1+sh)*Math.pow(Math.random(),1/sh); const d=sh-1/3,c=1/Math.sqrt(9*d); for(;;){let x,v; do{x=rNorm();v=1+c*x;}while(v<=0); v=v*v*v; const u=Math.random(); if(u<1-0.0331*x*x*x*x) return d*v; if(Math.log(u)<0.5*x*x+d*(1-v+Math.log(v))) return d*v;} }
function rBeta(a,b){ const ga=rGamma(a),gb=rGamma(b); return ga/(ga+gb); }
function rLogN(mean,std){ const mu=Math.log(mean*mean/Math.sqrt(std*std+mean*mean)),si=Math.sqrt(Math.log(1+(std/mean)**2)); return Math.exp(rNorm(mu,si)); }

/* ─── SIMULATION ENGINE ──────────────────────────────────────────────────── */
function runSim(p, N=10000, days=90){
  const paths=[]; const pE=Math.floor(N/80); const results=[];
  for(let s=0;s<N;s++){
    const sPass=Math.max(0.02,Math.min(0.45,rBeta(p.passRate*80,(1-p.passRate)*80)));
    const sPay =Math.max(0.003,Math.min(0.25,rBeta(p.payoutProb*40,(1-p.payoutProb)*40)));
    const sCorr=Math.max(0,Math.min(0.95,p.corr+rNorm(0,0.10)));
    let res=p.reserve,rev=0,pay=0,funded=0,maxDD=0,peak=res,ruined=false;
    const floor=p.reserve*p.emer; const snaps=[res];
    for(let d=0;d<days;d++){
      const sales=Math.max(0,Math.round(rNorm(p.salesDay,p.salesDay*0.5)));
      for(let i=0;i<sales;i++){
        const price=Math.random()<p.mix5k?p.p5k:p.p10k;
        const net=price*(1-p.aff-p.pmnt); res+=net; rev+=net;
        if(Math.random()<sPass){
          funded++;
          if(Math.random()<sPay){
            const shock=Math.random()<sCorr?Math.max(0.5,rNorm(1.6,0.4)):1;
            const payout=Math.max(30,rLogN(p.avgPay,p.payStd)*shock);
            res-=payout; pay+=payout;
          }
        }
      }
      if(Math.random()<p.fraud){ const fl=rLogN(280,140); res-=fl; pay+=fl; }
      res-=p.opDay;
      if(res>peak) peak=res;
      const dd=(peak-res)/Math.max(1,peak); if(dd>maxDD) maxDD=dd;
      if(res<floor&&!ruined) ruined=true;
      if(s%pE===0) snaps.push(res);
    }
    if(s%pE===0) paths.push(snaps);
    results.push({res,rev,pay,funded,maxDD,ruined});
  }
  results.sort((a,b)=>a.res-b.res);
  const pRuin=results.filter(r=>r.ruined).length/N;
  const pct=q=>results[Math.floor(N*q)].res;
  const avg=k=>results.reduce((a,r)=>a+r[k],0)/N;
  const hMin=Math.min(results[0].res,-100),hMax=results[N-1].res,bw=(hMax-hMin)/30;
  const hist=Array(30).fill(0);
  results.forEach(r=>{const i=Math.min(29,Math.floor((r.res-hMin)/bw));hist[i]++;});
  return {pRuin,p5:pct(.05),p10:pct(.10),p25:pct(.25),p50:pct(.50),p75:pct(.75),p90:pct(.90),p95:pct(.95),avgRev:avg("rev"),avgPay:avg("pay"),avgFunded:avg("funded"),avgDD:avg("maxDD"),worst:results[0].res,best:results[N-1].res,hist,hMin,bw,paths,N};
}

function calcUE(p){
  const calc=price=>{ const net=price*(1-p.aff-p.pmnt),liab=p.passRate*p.payoutProb*p.avgPay*p.split,infra=1.5,contrib=net-liab-infra; return {price,net,liab,infra,contrib,margin:contrib/price}; };
  return {p5k:calc(p.p5k),p10k:calc(p.p10k)};
}
function calcRL(p,sim){
  const emg=p.reserve*p.emer,pay=p.reserve*0.45,ops=p.reserve*0.20,gro=Math.max(0,p.reserve-emg-pay-ops);
  const stress=sim.avgPay*2.8,cov=(emg+pay)/Math.max(1,stress);
  return {emg,pay,ops,gro,stress,cov,exp:sim.avgPay};
}

/* ─── PATH CANVAS ────────────────────────────────────────────────────────── */
function PathCanvas({paths,reserve}){
  const ref=useRef(null);
  useEffect(()=>{
    if(!paths?.length||!ref.current) return;
    const canvas=ref.current,ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;
    ctx.clearRect(0,0,W,H);
    let minV=Infinity,maxV=-Infinity;
    paths.forEach(p=>p.forEach(v=>{if(v<minV)minV=v;if(v>maxV)maxV=v;}));
    minV=Math.min(minV,0); const range=maxV-minV||1;
    const toX=d=>(d/(paths[0].length-1))*W, toY=v=>H-((v-minV)/range)*(H*.88)-H*.06;
    ctx.strokeStyle=B.faint; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(0,toY(0)); ctx.lineTo(W,toY(0)); ctx.stroke();
    ctx.strokeStyle=B.border; ctx.setLineDash([3,6]);
    ctx.beginPath(); ctx.moveTo(0,toY(reserve)); ctx.lineTo(W,toY(reserve)); ctx.stroke();
    ctx.setLineDash([]);
    paths.forEach(path=>{
      const fv=path[path.length-1];
      ctx.strokeStyle=fv<0?B.red:fv<reserve?B.gold:B.green;
      ctx.globalAlpha=0.15; ctx.lineWidth=1; ctx.beginPath();
      path.forEach((v,i)=>i===0?ctx.moveTo(toX(i),toY(v)):ctx.lineTo(toX(i),toY(v))); ctx.stroke();
    });
    ctx.globalAlpha=1;
    const len=paths[0]?.length||0;
    const meds=Array(len).fill(0).map((_,i)=>{const vs=paths.map(p=>p[i]).sort((a,b)=>a-b);return vs[Math.floor(vs.length/2)];});
    ctx.strokeStyle=B.violetMid; ctx.lineWidth=2.5;
    ctx.beginPath(); meds.forEach((v,i)=>i===0?ctx.moveTo(toX(i),toY(v)):ctx.lineTo(toX(i),toY(v))); ctx.stroke();
  },[paths,reserve]);
  return <canvas ref={ref} width={600} height={190} style={{width:"100%",height:190,display:"block"}}/>;
}

/* ─── UI ATOMS ───────────────────────────────────────────────────────────── */
const Chip=({children,color=B.violetMid})=>(
  <span style={{display:"inline-block",padding:"2px 9px",borderRadius:2,background:color+"22",border:`1px solid ${color}55`,color,fontSize:10,fontFamily:B.mono,fontWeight:700,letterSpacing:"0.08em"}}>{children}</span>
);
const StatRow=({label,value,color=B.offWhite,sub,formula})=>(
  <div style={{padding:"9px 0",borderBottom:`1px solid ${B.border}`}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline"}}>
      <span style={{fontSize:12,color:B.muted}}>{label}</span>
      <span style={{fontSize:13,fontWeight:600,color,fontFamily:B.mono}}>{value}</span>
    </div>
    {(sub||formula)&&<div style={{fontSize:10,color:B.faint,fontFamily:B.mono,marginTop:2}}>{formula||sub}</div>}
  </div>
);
const KCard=({label,value,color=B.white,tag,sub,small})=>(
  <div style={{background:B.faint,border:`1px solid ${B.border}`,borderRadius:8,padding:"18px 20px",position:"relative"}}>
    {tag&&<div style={{position:"absolute",top:12,right:12}}><Chip color={color}>{tag}</Chip></div>}
    <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:8}}>{label}</div>
    <div style={{fontSize:small?18:26,fontWeight:700,color,fontFamily:B.mono,letterSpacing:"-0.02em",lineHeight:1}}>{value}</div>
    {sub&&<div style={{fontSize:11,color:B.muted,marginTop:6}}>{sub}</div>}
  </div>
);

/* ─── EXPLANATION BOX ────────────────────────────────────────────────────── */
const ExplainBox=({title,children})=>(
  <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderLeft:`3px solid ${B.violetMid}`,borderRadius:6,padding:"14px 16px",marginBottom:20}}>
    <div style={{fontSize:10,color:B.violetMid,fontFamily:B.mono,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:6}}>{title}</div>
    <div style={{fontSize:12,color:B.muted,lineHeight:1.7,fontFamily:B.sans}}>{children}</div>
  </div>
);

/* ─── FORMULA BOX ────────────────────────────────────────────────────────── */
const Formula=({children})=>(
  <div style={{background:B.black,border:`1px solid ${B.border}`,borderRadius:4,padding:"10px 14px",marginTop:8,marginBottom:8,fontFamily:B.mono,fontSize:11,color:B.gold,lineHeight:1.8}}>{children}</div>
);

/* ─── SLIDER ─────────────────────────────────────────────────────────────── */
function Slider({label,min,max,step,value,onChange,fmt,formula}){
  return (
    <div style={{marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
        <span style={{fontSize:11,color:B.muted}}>{label}</span>
        <span style={{fontSize:11,color:B.violetMid,fontFamily:B.mono,fontWeight:700}}>{fmt?fmt(value):value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))} style={{width:"100%",accentColor:B.violetMid,cursor:"pointer",height:3}}/>
      {formula&&<div style={{fontSize:10,color:B.faint,fontFamily:B.mono,marginTop:3}}>{formula}</div>}
    </div>
  );
}

const Divider=()=><div style={{height:1,background:B.border,margin:"20px 0"}}/>;

/* ─── PARAM CONTROLS ─────────────────────────────────────────────────────── */
function ParamControls({p,sp}){
  return (
    <>
      <div style={{fontSize:10,color:B.violetMid,fontFamily:B.mono,marginBottom:10,fontWeight:700,letterSpacing:"0.06em"}}>RESERVE</div>
      <Slider label="Starting reserve" min={5000} max={50000} step={500} value={p.reserve} onChange={sp("reserve")} fmt={fdF}/>
      <Slider label="Emergency floor %" min={0.10} max={0.50} step={0.01} value={p.emer} onChange={sp("emer")} fmt={v=>fp(v,0)} formula={`floor = reserve × ${fp(p.emer,0)}`}/>
      <Divider/>
      <div style={{fontSize:10,color:B.violetMid,fontFamily:B.mono,marginBottom:10,fontWeight:700,letterSpacing:"0.06em"}}>SALES</div>
      <Slider label="Sales per day" min={1} max={25} step={1} value={p.salesDay} onChange={sp("salesDay")} fmt={v=>`${v}/day`}/>
      <Slider label="5k price" min={29} max={120} step={1} value={p.p5k} onChange={sp("p5k")} fmt={v=>`$${v}`}/>
      <Slider label="10k price" min={49} max={200} step={1} value={p.p10k} onChange={sp("p10k")} fmt={v=>`$${v}`}/>
      <Slider label="5k mix" min={0.2} max={0.9} step={0.01} value={p.mix5k} onChange={sp("mix5k")} fmt={v=>fp(v,0)}/>
      <Divider/>
      <div style={{fontSize:10,color:B.violetMid,fontFamily:B.mono,marginBottom:10,fontWeight:700,letterSpacing:"0.06em"}}>RISK</div>
      <Slider label="Pass rate" min={0.02} max={0.40} step={0.005} value={p.passRate} onChange={sp("passRate")} fmt={v=>fp(v,1)}/>
      <Slider label="Payout probability" min={0.005} max={0.15} step={0.005} value={p.payoutProb} onChange={sp("payoutProb")} fmt={v=>fp(v,1)}/>
      <Slider label="Avg payout" min={50} max={1000} step={10} value={p.avgPay} onChange={sp("avgPay")} fmt={v=>`$${v}`}/>
      <Slider label="Payout std dev" min={50} max={800} step={10} value={p.payStd} onChange={sp("payStd")} fmt={v=>`$${v}`}/>
      <Slider label="Trader correlation" min={0} max={0.85} step={0.05} value={p.corr} onChange={sp("corr")} fmt={v=>fp(v,0)}/>
      <Divider/>
      <div style={{fontSize:10,color:B.violetMid,fontFamily:B.mono,marginBottom:10,fontWeight:700,letterSpacing:"0.06em"}}>COSTS</div>
      <Slider label="Affiliate %" min={0} max={0.30} step={0.01} value={p.aff} onChange={sp("aff")} fmt={v=>fp(v,0)}/>
      <Slider label="Payment cost %" min={0.01} max={0.08} step={0.005} value={p.pmnt} onChange={sp("pmnt")} fmt={v=>fp(v,1)}/>
      <Slider label="Op cost/day" min={0} max={100} step={5} value={p.opDay} onChange={sp("opDay")} fmt={v=>`$${v}`}/>
      <Slider label="Fraud frequency" min={0} max={0.03} step={0.001} value={p.fraud} onChange={sp("fraud")} fmt={v=>fp(v,2)}/>
      <Slider label="Profit split" min={0.5} max={0.95} step={0.05} value={p.split} onChange={sp("split")} fmt={v=>fp(v,0)}/>
    </>
  );
}

/* ─── MOBILE DRAWER ──────────────────────────────────────────────────────── */
function MobileDrawer({open,onClose,p,sp}){
  if(!open) return null;
  return (
    <div style={{position:"fixed",inset:0,zIndex:200,display:"flex"}}>
      <div onClick={onClose} style={{position:"absolute",inset:0,background:"#00000080"}}/>
      <div style={{position:"relative",width:260,background:B.violet,borderRight:`1px solid ${B.border}`,overflowY:"auto",padding:"20px 16px",zIndex:1}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
          <span style={{fontSize:11,color:B.muted,fontFamily:B.mono,letterSpacing:"0.1em",textTransform:"uppercase"}}>Parameters</span>
          <button onClick={onClose} style={{background:"none",border:"none",color:B.muted,cursor:"pointer",fontSize:20,lineHeight:1}}>×</button>
        </div>
        <ParamControls p={p} sp={sp}/>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION: OVERVIEW
══════════════════════════════════════════════════════════════════════════ */
function Overview({sim,rl,p}){
  const rc=sim.pRuin<0.01?B.green:sim.pRuin<0.05?B.gold:B.red;
  const cc=rl.cov>=2?B.green:rl.cov>=1.5?B.gold:B.red;
  const alerts=[];
  if(sim.pRuin>0.05) alerts.push({c:B.red,t:"Probability of ruin exceeds 5% — pause new funded accounts and review pricing."});
  else if(sim.pRuin>0.01) alerts.push({c:B.gold,t:"Ruin probability above 1% target — monitor reserve and correlation closely."});
  if(rl.cov<1.5) alerts.push({c:B.red,t:"Reserve coverage below 1.5× — do not expand product catalogue."});
  if(sim.p5<0) alerts.push({c:B.red,t:"Tail scenario produces negative reserve — stress reserve is insufficient."});
  if(sim.avgDD>0.35) alerts.push({c:B.gold,t:"Average drawdown exceeds 35% — review payout policy."});
  if(!alerts.length) alerts.push({c:B.green,t:"All primary risk metrics within target thresholds."});
  return (
    <div>
      <ExplainBox title="How this section works">
        The Overview aggregates outputs from all four layers beneath it: the Monte Carlo simulation (10,000 runs × 90 days),
        the reserve model (4-layer allocation), the unit economics engine, and the risk engine. Every number here is a
        derived output — nothing is hardcoded. Change any parameter on the left and re-run to see the full system update.
        The risk alerts fire automatically when computed metrics cross predefined thresholds.
      </ExplainBox>

      <div style={{marginBottom:20}}>
        {alerts.map((a,i)=>(
          <div key={i} style={{display:"flex",gap:10,alignItems:"flex-start",padding:"11px 16px",background:a.c+"11",border:`1px solid ${a.c}33`,borderRadius:6,marginBottom:6}}>
            <div style={{width:6,height:6,borderRadius:"50%",background:a.c,marginTop:5,flexShrink:0}}/>
            <span style={{fontSize:12,color:B.offWhite,lineHeight:1.6}}>{a.t}</span>
          </div>
        ))}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:10,marginBottom:10}}>
        <KCard label="Probability of Ruin" value={fp(sim.pRuin,2)} color={rc} tag={sim.pRuin<0.01?"ON TARGET":"ABOVE TARGET"} sub="Target < 1.00% · 10k sims"/>
        <KCard label="Reserve Coverage" value={`${rl.cov.toFixed(2)}×`} color={cc} tag={rl.cov>=2?"HEALTHY":"CAUTION"} sub="vs stress liability · Target ≥ 2×"/>
        <KCard label="Median Reserve (90d)" value={fd(sim.p50)} color={sim.p50>p.reserve?B.green:B.gold} sub={`From ${fdF(p.reserve)} start`}/>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(130px,1fr))",gap:10,marginBottom:20}}>
        <KCard label="Exp. Revenue" value={fd(sim.avgRev)} color={B.green} sub="90d avg net" small/>
        <KCard label="Exp. Payouts" value={fd(sim.avgPay)} color={B.gold} sub="90d avg" small/>
        <KCard label="Avg Funded" value={fn(sim.avgFunded,0)} color={B.offWhite} sub={`${fp(p.passRate)} pass rate`} small/>
        <KCard label="Avg Max Drawdown" value={fp(sim.avgDD)} color={sim.avgDD>0.3?B.red:B.gold} sub="Reserve peak→trough" small/>
      </div>

      <Divider/>
      <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:12}}>Reserve Allocation</div>
      <ExplainBox title="How reserve layers are calculated">
        Total reserve is split into 4 fixed-ratio buckets. Emergency = {fp(p.emer,0)} of starting reserve (your slider).
        Payout = 45%. Operating = 20%. Growth = remainder. The coverage ratio = (Emergency + Payout) ÷ Stress Liability,
        where Stress Liability = average simulated payouts × 2.8 (a conservative multiplier for tail scenarios).
      </ExplainBox>
      <div style={{display:"flex",height:10,borderRadius:3,overflow:"hidden",marginBottom:14}}>
        {[{v:rl.emg,c:B.red},{v:rl.pay,c:B.gold},{v:rl.ops,c:B.blue},{v:rl.gro,c:B.green}].map((s,i)=>(
          <div key={i} style={{width:`${(s.v/p.reserve)*100}%`,background:s.c}}/>
        ))}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8}}>
        {[
          ["Emergency",rl.emg,B.red,`${fp(p.emer,0)} of reserve · untouchable`],
          ["Payout",rl.pay,B.gold,"45% of reserve · funds withdrawals"],
          ["Operating",rl.ops,B.blue,"20% of reserve · day-to-day"],
          ["Growth",rl.gro,B.green,"Remainder · marketing & expansion"],
        ].map(([label,val,color,desc])=>(
          <div key={label} style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:6,padding:"12px 14px"}}>
            <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:6}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:color}}/>
              <span style={{fontSize:10,color:B.muted,textTransform:"uppercase",letterSpacing:"0.05em"}}>{label}</span>
            </div>
            <div style={{fontSize:16,fontWeight:700,color,fontFamily:B.mono}}>{fdF(val)}</div>
            <div style={{fontSize:10,color:B.faint,marginTop:3}}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION: MONTE CARLO
══════════════════════════════════════════════════════════════════════════ */
function MonteCarlo({sim,p}){
  const rc=sim.pRuin<0.01?B.green:sim.pRuin<0.05?B.gold:B.red;
  return (
    <div>
      <ExplainBox title="What Monte Carlo simulation does and why">
        A deterministic spreadsheet gives you one answer based on fixed assumptions. Monte Carlo gives you a
        probability distribution of answers by running 10,000 independent 90-day scenarios, each with slightly
        different versions of reality (different pass rates, payout sizes, sales volumes, correlation shocks).
        The result is not a single forecast — it is a map of possible futures. Decision-making from distributions
        is more honest and more useful than decision-making from point estimates.
      </ExplainBox>

      <div style={{marginBottom:6,fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase"}}>Simulation Paths — 90-Day Reserve Evolution</div>
      <div style={{fontSize:11,color:B.muted,marginBottom:12}}>Each line = one simulated 90-day run. Purple = median trajectory across all paths.</div>
      <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"16px",marginBottom:20}}>
        <PathCanvas paths={sim.paths} reserve={p.reserve}/>
        <div style={{display:"flex",flexWrap:"wrap",gap:16,marginTop:10}}>
          {[["Ruin path (reserve < floor)",B.red],["Below starting reserve",B.gold],["Above starting reserve",B.green],["Median path",B.violetMid]].map(([l,c])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:5}}>
              <div style={{width:16,height:2,background:c,borderRadius:1}}/>
              <span style={{fontSize:10,color:B.muted}}>{l}</span>
            </div>
          ))}
        </div>
      </div>

      <ExplainBox title="How each simulation run works — step by step">
        For each of 10,000 runs: (1) Sample a pass rate from Beta distribution centered on your input — this adds
        realistic uncertainty so not every run uses exactly {fp(p.passRate)}. (2) Each day, draw daily sales from
        Normal(μ={p.salesDay}, σ={(p.salesDay*.5).toFixed(1)}). (3) For each sale, flip a weighted coin for 5k vs 10k,
        deduct affiliate+payment costs, add net revenue to reserve. (4) If trader passes evaluation, flip another coin
        against payout probability. If they pay out, draw payout size from Log-Normal distribution — this creates a
        realistic long right tail where most payouts are small but occasional ones are large. (5) Apply correlation
        shock: if a random draw &lt; correlation setting, multiply payout by ~1.6× to simulate market-driven clustering.
        (6) Check daily fraud event. (7) Deduct operating cost. (8) Record if reserve breached emergency floor (ruin).
      </ExplainBox>

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))",gap:12,marginBottom:12}}>
        <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
          <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:14}}>Percentile Outcomes — Final Reserve</div>
          {[
            ["5th — worst 5% of runs",sim.p5,"Bottom tail: stress scenario"],
            ["10th percentile",sim.p10,""],
            ["25th percentile",sim.p25,""],
            ["50th — median",sim.p50,"Expected case"],
            ["75th percentile",sim.p75,""],
            ["90th percentile",sim.p90,""],
            ["95th — best 5% of runs",sim.p95,"Upper tail"],
          ].map(([l,v,s])=>(
            <StatRow key={l} label={l} value={fdF(v)} color={v<0?B.red:v<p.reserve?B.gold:B.green} sub={s}/>
          ))}
        </div>
        <div>
          <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px",marginBottom:12}}>
            <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:14}}>Risk Summary</div>
            <StatRow label="Probability of ruin" value={fp(sim.pRuin,3)} color={rc} formula="runs where res < floor ÷ 10,000"/>
            <StatRow label="Worst single run" value={fdF(sim.worst)} color={B.red} formula="min(all final reserves)"/>
            <StatRow label="Best single run" value={fdF(sim.best)} color={B.green} formula="max(all final reserves)"/>
            <StatRow label="Avg max drawdown" value={fp(sim.avgDD)} color={sim.avgDD>0.3?B.red:B.gold} formula="avg((peak−trough)/peak) per run"/>
            <StatRow label="Net avg (rev − pay)" value={fdF(sim.avgRev-sim.avgPay)} color={B.offWhite}/>
          </div>
          <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
            <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:14}}>Distributions Used</div>
            <StatRow label="Pass rate" value="Beta(α,β)" sub={`α=${(p.passRate*80).toFixed(0)}, β=${((1-p.passRate)*80).toFixed(0)}`}/>
            <StatRow label="Payout prob" value="Beta(α,β)" sub={`α=${(p.payoutProb*40).toFixed(0)}, β=${((1-p.payoutProb)*40).toFixed(0)}`}/>
            <StatRow label="Payout size" value="Log-Normal" sub={`μ=$${p.avgPay}, σ=$${p.payStd} → long right tail`}/>
            <StatRow label="Daily sales" value="Normal" sub={`μ=${p.salesDay}/day, σ=${(p.salesDay*.5).toFixed(1)}`}/>
            <StatRow label="Correlation shock" value="Bernoulli" sub={`p=${fp(p.corr,0)} → ×1.6 payout multiplier`}/>
            <StatRow label="Fraud event" value="Bernoulli" sub={`p=${fp(p.fraud,2)}/day → Log-Normal loss`}/>
          </div>
        </div>
      </div>

      <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
        <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:4}}>Final Reserve Distribution — 10,000 Runs</div>
        <div style={{fontSize:11,color:B.muted,marginBottom:14}}>Histogram of final reserve values. Red bars = ruin scenarios. Gold = below start. Green = growth.</div>
        <svg width="100%" viewBox="0 0 600 90" preserveAspectRatio="none" style={{display:"block",height:90}}>
          {sim.hist.map((count,i)=>{
            const max=Math.max(...sim.hist),bH=(count/max)*82,mid=sim.hMin+(i+0.5)*sim.bw;
            return <rect key={i} x={i*(600/30)+1} y={90-bH} width={600/30-2} height={bH} fill={mid<0?B.red:mid<p.reserve?B.gold:B.green} opacity={0.8} rx={1}/>;
          })}
        </svg>
        <div style={{display:"flex",justifyContent:"space-between",marginTop:6}}>
          <span style={{fontSize:10,color:B.muted,fontFamily:B.mono}}>{fdF(sim.hMin)}</span>
          <span style={{fontSize:10,color:B.muted,fontFamily:B.mono,textAlign:"center"}}>← final reserve after 90 days →</span>
          <span style={{fontSize:10,color:B.muted,fontFamily:B.mono}}>{fdF(sim.hMin+30*sim.bw)}</span>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION: UNIT ECONOMICS
══════════════════════════════════════════════════════════════════════════ */
function UnitEcon({ue,p}){
  const d5k=p.salesDay*p.mix5k, d10k=p.salesDay*(1-p.mix5k);
  const dRev=d5k*ue.p5k.net+d10k*ue.p10k.net;
  const dLia=d5k*ue.p5k.liab+d10k*ue.p10k.liab;
  const dCon=dRev-dLia-p.opDay;
  return (
    <div>
      <ExplainBox title="The key insight: revenue is certain, liability is probabilistic">
        Every challenge sold creates two simultaneous financial events: (1) immediate deterministic cash inflow,
        and (2) a probabilistic future cash outflow. Most prop firms only track the inflow. This model tracks both.
        Expected Liability = P(pass) × P(payout | funded) × average payout × profit split. This is what a challenge
        actually costs the firm on a probabilistic basis — not what the challenge earns. Contribution = Net Revenue − Expected Liability − Infrastructure.
        A product is only viable if contribution is positive across many sales.
      </ExplainBox>

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))",gap:14,marginBottom:16}}>
        {[["5k Challenge",ue.p5k],["10k Challenge",ue.p10k]].map(([title,u])=>(
          <div key={title} style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"22px"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:18}}>
              <span style={{fontSize:14,fontWeight:700,color:B.white}}>{title}</span>
              <Chip color={u.margin>0.3?B.green:u.margin>0.1?B.gold:B.red}>{fp(u.margin)} margin</Chip>
            </div>
            {[
              ["Challenge price",`+${fdF(u.price)}`,B.green,`gross inflow per sale`],
              ["Affiliate + payment",`-${fdF(u.price*(p.aff+p.pmnt))}`,B.red,`price × (${fp(p.aff,0)} aff + ${fp(p.pmnt,1)} pmnt)`],
              ["= Net revenue",fdF(u.net),B.offWhite,`price × (1 − ${fp(p.aff+p.pmnt,1)})`],
              ["Expected payout liability",`-${fdF(u.liab)}`,B.red,`${fp(p.passRate)} × ${fp(p.payoutProb)} × $${p.avgPay} × ${fp(p.split,0)}`],
              ["Infrastructure alloc.",`-${fdF(u.infra)}`,B.muted,`fixed $1.50 per evaluation`],
            ].map(([l,v,c,s])=>(
              <StatRow key={l} label={l} value={v} color={c} formula={s}/>
            ))}
            <div style={{display:"flex",justifyContent:"space-between",paddingTop:14}}>
              <span style={{fontSize:13,fontWeight:600,color:B.white}}>Economic contribution</span>
              <span style={{fontSize:20,fontWeight:700,color:u.contrib>0?B.green:B.red,fontFamily:B.mono}}>{fdF(u.contrib)}</span>
            </div>
            <Formula>contribution = net_rev − exp_liability − infra{"\n"}= {fdF(u.net)} − {fdF(u.liab)} − {fdF(u.infra)} = {fdF(u.contrib)}</Formula>
          </div>
        ))}
      </div>

      <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"22px"}}>
        <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:8}}>Portfolio Economics — {fn(p.salesDay,0)} sales/day ({fp(p.mix5k,0)} 5k, {fp(1-p.mix5k,0)} 10k)</div>
        <Formula>daily_revenue = ({d5k.toFixed(1)} × ${ue.p5k.net.toFixed(0)}) + ({d10k.toFixed(1)} × ${ue.p10k.net.toFixed(0)}) = ${dRev.toFixed(0)}{"\n"}daily_liability = ({d5k.toFixed(1)} × ${ue.p5k.liab.toFixed(0)}) + ({d10k.toFixed(1)} × ${ue.p10k.liab.toFixed(0)}) = ${dLia.toFixed(0)}{"\n"}daily_contribution = ${dRev.toFixed(0)} − ${dLia.toFixed(0)} − ${p.opDay} (op) = ${dCon.toFixed(0)}</Formula>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(140px,1fr))",gap:8,marginTop:14}}>
          {[
            ["Daily net revenue",fdF(dRev),B.green],
            ["Daily exp. liability",fdF(dLia),B.gold],
            ["Daily op cost",fdF(p.opDay),B.red],
            ["Daily contribution",fdF(dCon),dCon>0?B.green:B.red],
            ["Monthly contribution",fdF(dCon*30),dCon>0?B.green:B.red],
            ["Months to 2× reserve",dCon>0?fn(p.reserve/(dCon*30),1):"∞",B.offWhite],
            ["Liability / revenue",fp(dLia/Math.max(1,dRev)),B.offWhite],
            ["Reserve runway",dCon>0?"Indefinite":fn(p.reserve/Math.max(.1,p.opDay-dCon),0)+" days",B.offWhite],
          ].map(([k,v,c])=>(
            <div key={k} style={{background:B.faint,borderRadius:6,padding:"12px 14px",border:`1px solid ${B.border}`}}>
              <div style={{fontSize:10,color:B.muted,textTransform:"uppercase",letterSpacing:"0.05em",marginBottom:6}}>{k}</div>
              <div style={{fontSize:15,fontWeight:700,color:c,fontFamily:B.mono}}>{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION: RISK ENGINE  (fully editable trader classes)
══════════════════════════════════════════════════════════════════════════ */
const DEFAULT_CLASSES = [
  {cls:"A — Casual",       share:0.55, mult:0.10, color:B.green,  desc:"Emotional, high-variance, fail quickly. Low reserve impact."},
  {cls:"B — Learning",     share:0.22, mult:0.40, color:B.blue,   desc:"Improving consistency, occasional profit, repeat purchases."},
  {cls:"C — Disciplined",  share:0.12, mult:1.00, color:B.gold,   desc:"Stable risk, repeatable process. Meaningful reserve impact."},
  {cls:"D — Professional", share:0.09, mult:2.80, color:"#F97316",desc:"Multiple payouts, long funded lifespan. High reserve impact."},
  {cls:"E — Outlier",      share:0.02, mult:9.00, color:B.red,    desc:"Rare. Disproportionately large payouts. Determines reserve sizing."},
];

function RiskEngine({sim,p,rl}){
  const [classes, setClasses] = useState(DEFAULT_CLASSES);
  const [showEdit, setShowEdit] = useState(false);

  const updateClass=(i,field,val)=>{
    const updated=[...classes];
    updated[i]={...updated[i],[field]:val};
    setClasses(updated);
  };

  const totalShare=classes.reduce((a,c)=>a+c.share,0);
  const fundedTotal=Math.round(sim.avgFunded);

  const enriched=classes.map(c=>({
    ...c,
    count:      Math.round(fundedTotal * (c.share/totalShare)),
    expPay:     c.mult * p.avgPay,
    totalLiab:  Math.round(fundedTotal * (c.share/totalShare)) * c.mult * p.avgPay,
  }));
  const totalLiab=enriched.reduce((a,c)=>a+c.totalLiab,0);

  return (
    <div>
      <ExplainBox title="How the risk engine models traders">
        Traders are not identical. Assuming they are produces dangerously inaccurate reserve estimates.
        The risk engine divides the funded trader population into 5 behavioural classes, each with a
        population share and a payout multiplier relative to the average payout. The multiplier encodes
        how much more (or less) a trader in that class is expected to withdraw compared to the average.
        Class E Outliers (2% of funded traders) have a 9× multiplier — meaning one outlier expects to
        pay out 9× the average. These traders alone determine whether reserve is sufficient.
        Edit the shares and multipliers below to model your own assumptions.
      </ExplainBox>

      {/* Trader class editor toggle */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
        <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase"}}>
          Trader Classes — {fundedTotal} avg funded traders per 90-day run
        </div>
        <button onClick={()=>setShowEdit(!showEdit)} style={{background:showEdit?B.violetMid:"transparent",border:`1px solid ${B.borderHi}`,borderRadius:4,color:B.offWhite,padding:"5px 12px",fontSize:11,fontFamily:B.mono,cursor:"pointer"}}>
          {showEdit?"Close Editor":"Edit Classes"}
        </button>
      </div>

      {showEdit&&(
        <div style={{background:B.violetDim,border:`1px solid ${B.borderHi}`,borderRadius:8,padding:"16px",marginBottom:16}}>
          <div style={{fontSize:10,color:B.muted,marginBottom:12,fontFamily:B.sans}}>
            Adjust population shares (must sum to 100%) and payout multipliers. Multiplier is relative to avg payout of ${p.avgPay}.
            Share total: <span style={{color:Math.abs(totalShare-1)<0.01?B.green:B.red,fontFamily:B.mono,fontWeight:700}}>{fp(totalShare,0)}</span>
          </div>
          {classes.map((c,i)=>(
            <div key={c.cls} style={{background:B.faint,border:`1px solid ${B.border}`,borderRadius:6,padding:"14px",marginBottom:8}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
                <div style={{width:8,height:8,borderRadius:"50%",background:c.color}}/>
                <span style={{fontSize:12,fontWeight:600,color:B.white}}>{c.cls}</span>
                <span style={{fontSize:10,color:B.muted,fontFamily:B.sans,marginLeft:4}}>{c.desc}</span>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                <div>
                  <div style={{fontSize:10,color:B.muted,marginBottom:4}}>Population share: <span style={{color:c.color,fontFamily:B.mono}}>{fp(c.share,0)}</span></div>
                  <input type="range" min={0.01} max={0.70} step={0.01} value={c.share}
                    onChange={e=>updateClass(i,"share",Number(e.target.value))}
                    style={{width:"100%",accentColor:c.color,height:3}}/>
                </div>
                <div>
                  <div style={{fontSize:10,color:B.muted,marginBottom:4}}>Payout multiplier: <span style={{color:c.color,fontFamily:B.mono}}>{c.mult.toFixed(2)}×</span> = <span style={{color:B.gold,fontFamily:B.mono}}>${(c.mult*p.avgPay).toFixed(0)}</span></div>
                  <input type="range" min={0.01} max={15} step={0.05} value={c.mult}
                    onChange={e=>updateClass(i,"mult",Number(e.target.value))}
                    style={{width:"100%",accentColor:c.color,height:3}}/>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Class population bar */}
      <div style={{display:"flex",height:10,borderRadius:3,overflow:"hidden",marginBottom:14}}>
        {enriched.map(c=><div key={c.cls} style={{width:`${(c.share/totalShare)*100}%`,background:c.color}}/>)}
      </div>

      {/* Main class table */}
      <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,overflow:"hidden",marginBottom:16}}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 60px 80px 80px 90px 90px",gap:0,padding:"10px 18px",background:B.faint,borderBottom:`1px solid ${B.border}`}}>
          {["Class","Count","Share","Mult","Exp Payout","Total Liab"].map(h=>(
            <span key={h} style={{fontSize:9,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase"}}>{h}</span>
          ))}
        </div>
        {enriched.map((c,i)=>(
          <div key={c.cls} style={{display:"grid",gridTemplateColumns:"1fr 60px 80px 80px 90px 90px",gap:0,padding:"12px 18px",borderBottom:i<enriched.length-1?`1px solid ${B.border}`:"none",background:i%2===0?"transparent":B.faint+"33"}}>
            <div style={{display:"flex",alignItems:"center",gap:7}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:c.color,flexShrink:0}}/>
              <span style={{fontSize:11,color:B.offWhite}}>{c.cls}</span>
            </div>
            <span style={{fontSize:11,color:B.muted,fontFamily:B.mono,textAlign:"right",paddingRight:8}}>{c.count}</span>
            <span style={{fontSize:11,color:B.muted,fontFamily:B.mono,textAlign:"right",paddingRight:8}}>{fp(c.share/totalShare,0)}</span>
            <span style={{fontSize:11,color:c.color,fontFamily:B.mono,textAlign:"right",paddingRight:8}}>{c.mult.toFixed(2)}×</span>
            <span style={{fontSize:11,color:B.offWhite,fontFamily:B.mono,textAlign:"right",paddingRight:8}}>{fdF(c.expPay)}</span>
            <span style={{fontSize:11,color:c.color,fontFamily:B.mono,textAlign:"right",fontWeight:700}}>{fd(c.totalLiab)}</span>
          </div>
        ))}
        <div style={{display:"grid",gridTemplateColumns:"1fr 60px 80px 80px 90px 90px",gap:0,padding:"12px 18px",background:B.faint,borderTop:`1px solid ${B.border}`}}>
          <span style={{fontSize:11,color:B.muted}}>Total</span>
          <span style={{fontSize:11,color:B.white,fontFamily:B.mono,textAlign:"right",paddingRight:8,fontWeight:700}}>{fundedTotal}</span>
          <span style={{fontSize:11,color:B.white,fontFamily:B.mono,textAlign:"right",paddingRight:8}}>{fp(totalShare,0)}</span>
          <span/>
          <span/>
          <span style={{fontSize:11,color:B.white,fontFamily:B.mono,textAlign:"right",fontWeight:700}}>{fd(totalLiab)}</span>
        </div>
      </div>

      <ExplainBox title="How trader count is calculated">
        avg_funded = average number of traders who passed evaluation across all 10,000 simulation runs.
        Currently: {fundedTotal} traders. Each class count = funded_total × (class_share ÷ total_share).
        Expected payout per class = class_multiplier × avg_payout_param (${p.avgPay}).
        Total liability per class = count × expected_payout.
        The outlier class (E) is small in count but dominates total liability — this is tail risk in practice.
      </ExplainBox>
      <Formula>
        count[class] = {fundedTotal} × share[class]{"\n"}
        exp_payout[class] = mult[class] × ${p.avgPay}{"\n"}
        total_liability[class] = count × exp_payout{"\n"}
        {"\n"}
        Class E example: {enriched[4]?.count} traders × ${enriched[4]?.expPay.toFixed(0)} = {fd(enriched[4]?.totalLiab)} liability
      </Formula>

      <Divider/>

      {/* Concentration bars */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))",gap:12,marginBottom:16}}>
        <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
          <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:12}}>Liability Concentration</div>
          <div style={{fontSize:11,color:B.muted,marginBottom:12}}>What % of total expected liability comes from each class?</div>
          {enriched.map(c=>(
            <div key={c.cls} style={{marginBottom:10}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
                <span style={{fontSize:11,color:B.muted}}>{c.cls}</span>
                <span style={{fontSize:11,color:c.color,fontFamily:B.mono,fontWeight:700}}>
                  {totalLiab>0?fp(c.totalLiab/totalLiab):"—"} · {fd(c.totalLiab)}
                </span>
              </div>
              <div style={{height:5,background:B.border,borderRadius:2}}>
                <div style={{height:"100%",width:`${totalLiab>0?(c.totalLiab/totalLiab)*100:0}%`,background:c.color,borderRadius:2}}/>
              </div>
            </div>
          ))}
        </div>

        <div style={{display:"flex",flexDirection:"column",gap:12}}>
          <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
            <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:12}}>Portfolio Risk Factors</div>
            <StatRow label="Trader correlation" value={fp(p.corr)} color={p.corr>0.4?B.red:p.corr>0.2?B.gold:B.green} formula="when market trends, profitable traders cluster"/>
            <StatRow label="Fraud freq (daily)" value={fp(p.fraud,2)} color={p.fraud>0.01?B.red:B.green} formula="Bernoulli trial each day → Log-Normal loss"/>
            <StatRow label="Stress liability (2.8×)" value={fdF(rl.stress)} color={B.gold} formula="avg_sim_payouts × 2.8 conservative multiplier"/>
            <StatRow label="Reserve coverage" value={`${rl.cov.toFixed(2)}×`} color={rl.cov>=2?B.green:B.red} formula="(emergency + payout pool) ÷ stress_liability"/>
          </div>
          <div style={{background:B.violetDim,border:`1px solid ${B.border}`,borderRadius:8,padding:"20px"}}>
            <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:12}}>Product Gate</div>
            {[{prod:"5k",reqs:["Cov > 1.5×","P(Ruin) < 5%"],met:rl.cov>=1.5&&sim.pRuin<0.05},{prod:"10k",reqs:["Cov > 1.5×","P(Ruin) < 5%"],met:rl.cov>=1.5&&sim.pRuin<0.05},{prod:"25k",reqs:["Cov > 2.5×","P(Ruin) < 1%"],met:rl.cov>=2.5&&sim.pRuin<0.01},{prod:"50k",reqs:["Cov > 4×","P(Ruin) < 0.5%"],met:rl.cov>=4&&sim.pRuin<0.005}].map(pp=>(
              <div key={pp.prod} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"7px 0",borderBottom:`1px solid ${B.border}`}}>
                <span style={{fontSize:12,fontFamily:B.mono,color:B.white}}>{pp.prod}</span>
                <div style={{display:"flex",alignItems:"center",gap:8}}>
                  <span style={{fontSize:10,color:B.muted}}>{pp.reqs.join(" · ")}</span>
                  <Chip color={pp.met?B.green:B.faint}>{pp.met?"LIVE":"LOCKED"}</Chip>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   MAIN APP
══════════════════════════════════════════════════════════════════════════ */
const TABS=["Overview","Monte Carlo","Unit Economics","Risk Engine"];

export default function App(){
  const [tab,setTab]=useState("Overview");
  const [busy,setBusy]=useState(false);
  const [sim,setSim]=useState(null);
  const [drawer,setDrawer]=useState(false);

  const [p,setP]=useState({
    reserve:10000,emer:0.25,salesDay:3,p5k:59,p10k:99,mix5k:0.62,
    passRate:0.128,payoutProb:0.030,avgPay:280,payStd:200,
    split:0.80,corr:0.25,aff:0.15,pmnt:0.04,opDay:15,fraud:0.005,
  });
  const sp=k=>v=>setP(prev=>({...prev,[k]:v}));
  const go=useCallback(()=>{setBusy(true);setTimeout(()=>{setSim(runSim(p,10000,90));setBusy(false);},80);},[p]);
  useEffect(()=>{go();},[]);

  const ue=useMemo(()=>calcUE(p),[p]);
  const rl=useMemo(()=>sim?calcRL(p,sim):null,[p,sim]);

  return (
    <div style={{background:B.black,minHeight:"100vh",color:B.offWhite,fontFamily:B.sans,fontSize:13}}>
      {/* TOPBAR */}
      <div style={{borderBottom:`1px solid ${B.border}`,background:B.violet,position:"sticky",top:0,zIndex:100}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 16px",height:54}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <button onClick={()=>setDrawer(true)} style={{background:"none",border:"none",color:B.muted,cursor:"pointer",padding:4,display:"flex",alignItems:"center"}} aria-label="Parameters">
              <svg width="18" height="14" viewBox="0 0 18 14" fill="none">
                <rect x="0" y="0"   width="18" height="2" rx="1" fill={B.muted}/>
                <rect x="0" y="6"   width="18" height="2" rx="1" fill={B.muted}/>
                <rect x="0" y="12"  width="18" height="2" rx="1" fill={B.muted}/>
              </svg>
            </button>
            <SkwidMark size={24}/>
            <div>
              <div style={{fontSize:14,fontWeight:800,color:B.white,letterSpacing:"0.06em",fontFamily:B.mono}}>SKWID</div>
              <div style={{fontSize:9,color:B.muted,letterSpacing:"0.1em"}}>RISK PLATFORM · RESEARCH STAGE</div>
            </div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            {sim&&<span style={{fontSize:10,color:B.muted,fontFamily:B.mono}}>10k sims · 90d</span>}
            <button onClick={go} disabled={busy} style={{background:busy?"transparent":B.violetMid,border:`1px solid ${busy?B.border:B.violetMid}`,borderRadius:4,color:busy?B.muted:B.white,padding:"7px 14px",fontSize:11,fontWeight:700,cursor:busy?"not-allowed":"pointer",fontFamily:B.mono,letterSpacing:"0.06em"}}>
              {busy?"COMPUTING…":"RUN SIM"}
            </button>
          </div>
        </div>
        <div style={{display:"flex",overflowX:"auto",WebkitOverflowScrolling:"touch",scrollbarWidth:"none",padding:"0 8px"}}>
          {TABS.map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{background:"none",border:"none",cursor:"pointer",whiteSpace:"nowrap",padding:"0 14px",height:38,fontSize:12,fontWeight:tab===t?700:400,color:tab===t?B.white:B.muted,borderBottom:tab===t?`2px solid ${B.violetMid}`:"2px solid transparent",flexShrink:0}}>
              {t}
            </button>
          ))}
        </div>
      </div>

      <div style={{display:"flex",minHeight:"calc(100vh - 92px)"}}>
        {/* Desktop sidebar */}
        <div style={{width:230,flexShrink:0,borderRight:`1px solid ${B.border}`,padding:"20px 16px",overflowY:"auto",background:B.violet,display:"none"}} id="desktop-sidebar">
          <div style={{fontSize:10,color:B.muted,fontFamily:B.mono,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:18}}>Parameters</div>
          <ParamControls p={p} sp={sp}/>
        </div>

        <MobileDrawer open={drawer} onClose={()=>setDrawer(false)} p={p} sp={sp}/>

        {/* Content */}
        <div style={{flex:1,padding:"24px 16px",overflowY:"auto",maxWidth:"100%"}}>
          <div style={{marginBottom:20}}>
            <div style={{fontSize:9,color:B.muted,fontFamily:B.mono,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:4}}>{
              tab==="Overview"?"Company Health · Reserve · Risk Alerts":
              tab==="Monte Carlo"?"10,000 Simulations · 90-Day Horizon · Probability Distributions":
              tab==="Unit Economics"?"Per-Product Economics · Contribution Margin · Portfolio":
              "Trader Classification · Liability Model · Concentration · Product Gates"
            }</div>
            <div style={{fontSize:22,fontWeight:800,color:B.white,letterSpacing:"-0.01em",fontFamily:B.mono}}>{tab}</div>
          </div>

          {!sim?(
            <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:300,color:B.muted,fontFamily:B.mono}}>Computing…</div>
          ):(
            <>
              {tab==="Overview"       && <Overview sim={sim} rl={rl} p={p}/>}
              {tab==="Monte Carlo"    && <MonteCarlo sim={sim} p={p}/>}
              {tab==="Unit Economics" && <UnitEcon ue={ue} p={p}/>}
              {tab==="Risk Engine"    && <RiskEngine sim={sim} p={p} rl={rl}/>}
            </>
          )}

          <div style={{marginTop:40,paddingTop:20,borderTop:`1px solid ${B.border}`,display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:8}}>
            <div style={{display:"flex",alignItems:"center",gap:8}}>
              <SkwidMark size={14}/>
              <span style={{fontSize:10,color:B.faint,fontFamily:B.mono}}>SKWID · Research Stage · {new Date().getFullYear()}</span>
            </div>
            <span style={{fontSize:10,color:B.faint,fontFamily:B.mono}}>10,000 Monte Carlo · 90-day horizon</span>
          </div>
        </div>
      </div>

      <style>{`
        *{box-sizing:border-box;}
        ::-webkit-scrollbar{width:4px;height:4px;}
        ::-webkit-scrollbar-track{background:${B.black};}
        ::-webkit-scrollbar-thumb{background:${B.border};border-radius:2px;}
        input[type=range]{-webkit-appearance:none;background:${B.border};border-radius:2px;height:3px;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:${B.violetMid};cursor:pointer;border:2px solid ${B.white};}
        @media(min-width:768px){#desktop-sidebar{display:block!important;}}
      `}</style>
    </div>
  );
}
