import { createServer, type ServerResponse } from "node:http";
import { resolve } from "node:path";

import { readClassifierStatuses } from "./status.js";

export { readClassifierStatuses } from "./status.js";
export type {
  ClassifierStatus,
  ReadClassifierStatusesOptions,
} from "./status.js";

export interface StartClassifiersUiOptions {
  readonly dataDirectory?: string;
  readonly host?: string;
  readonly port?: number;
}

export interface ClassifiersUiServer {
  readonly url: string;
  close(): Promise<void>;
}

export async function startClassifiersUi(
  options: StartClassifiersUiOptions = {},
): Promise<ClassifiersUiServer> {
  const dataDirectory = resolve(
    options.dataDirectory ?? process.env.SWAPAI_DATA_DIRECTORY ?? ".swapai",
  );
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 4789;
  if (!Number.isSafeInteger(port) || port < 0 || port > 65_535) {
    throw new TypeError("port must be a whole number between 0 and 65535");
  }

  const server = createServer((request, response) => {
    setHeaders(response);
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" });
      response.end();
      return;
    }
    const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
    if (pathname === "/health") {
      sendJson(response, 200, { service: "swapai-classifiers-ui", status: "ok" }, request.method);
      return;
    }
    if (pathname === "/api/classifiers") {
      try {
        sendJson(response, 200, {
          generatedAt: Date.now(),
          classifiers: readClassifierStatuses({ dataDirectory }),
        }, request.method);
      } catch (error) {
        sendJson(response, 500, {
          error: error instanceof Error ? error.message : "Could not read classifier status",
        }, request.method);
      }
      return;
    }
    if (pathname === "/") {
      const body = request.method === "HEAD" ? undefined : PAGE;
      response.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": Buffer.byteLength(PAGE),
      });
      response.end(body);
      return;
    }
    response.writeHead(404);
    response.end();
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => {
      server.off("error", reject);
      resolveListen();
    });
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    server.close();
    throw new Error("Classifier monitor did not start on a TCP port");
  }
  const displayHost = host.includes(":") ? `[${host}]` : host;
  return {
    url: `http://${displayHost}:${address.port}/`,
    close: () => new Promise<void>((resolveClose, reject) => {
      server.close((error) => error === undefined ? resolveClose() : reject(error));
    }),
  };
}

function setHeaders(response: ServerResponse): void {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
  );
}

function sendJson(
  response: ServerResponse,
  status: number,
  value: unknown,
  method: string | undefined,
): void {
  const serialized = JSON.stringify(value);
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(serialized),
  });
  response.end(method === "HEAD" ? undefined : serialized);
}

const PAGE = String.raw`<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SwapAI classifiers</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0b0c; --panel:#121214; --line:#26262a; --muted:#929299; --text:#f5f5f4; --yellow:#f7e45c; --green:#62d692; --blue:#77b7ff; --red:#ff7c7c; }
    * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { position:sticky; top:0; z-index:2; display:flex; align-items:center; justify-content:space-between; padding:18px 28px; background:rgba(11,11,12,.9); border-bottom:1px solid var(--line); backdrop-filter:blur(14px); }
    .brand { display:flex; gap:12px; align-items:center; font-weight:700; font-size:16px } .mark { width:22px; height:22px; border-radius:50%; background:linear-gradient(135deg,#171719 22%,var(--yellow) 72%); box-shadow:0 0 20px #f7e45c33; }
    .updated { color:var(--muted); font-size:12px } main { width:min(1180px,calc(100% - 40px)); margin:0 auto; padding:54px 0 80px; }
    h1 { margin:0; font-size:clamp(30px,5vw,48px); letter-spacing:-.045em } .intro { color:var(--muted); font-size:17px; margin:8px 0 32px }
    .summary { display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap } .summary span,.pill { border:1px solid var(--line); background:#171719; border-radius:999px; padding:5px 10px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:14px } .card { background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:20px; box-shadow:0 12px 45px #0003; }
    .card-head { display:flex; justify-content:space-between; gap:15px; align-items:flex-start; margin-bottom:20px } h2 { margin:0; font-size:18px; letter-spacing:-.02em; overflow-wrap:anywhere } .type { color:var(--muted); font-size:12px; margin-top:3px }
    .states { display:flex; flex-wrap:wrap; gap:6px; justify-content:flex-end } .pill { padding:3px 8px; font-size:11px; color:var(--muted) } .pill.on { color:var(--green); border-color:#62d69255; background:#62d69210 } .pill.training { color:var(--blue); border-color:#77b7ff55; background:#77b7ff10 }
    dl { display:grid; grid-template-columns:1fr auto; margin:0; gap:0 } dt,dd { margin:0; padding:11px 0; border-top:1px solid var(--line) } dt { color:var(--muted) } dd { font-variant-numeric:tabular-nums; font-weight:600; text-align:right } .good { color:var(--green) } .bad { color:var(--red) }
    .bar { grid-column:1/-1; height:5px; padding:0; margin:-3px 0 10px; border:0; border-radius:5px; background:#252529; overflow:hidden } .bar i { display:block; height:100%; background:var(--yellow); border-radius:inherit }
    .facts { margin-top:20px; display:grid; gap:14px } .fact { border-top:1px solid var(--line); padding-top:13px } .fact h3 { margin:0 0 8px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em }
    .fact ul { list-style:none; padding:0; margin:0; display:grid; gap:6px } .fact li,.run { background:#171719; border-radius:8px; padding:8px 10px; overflow-wrap:anywhere } .meta { color:var(--muted); font-size:12px } .run { display:grid; gap:8px; margin-top:8px } .run-head { display:flex; justify-content:space-between; gap:12px } .failure { color:var(--red) }
    .empty,.error { border:1px dashed var(--line); border-radius:15px; padding:42px; color:var(--muted); text-align:center } .error { color:var(--red) }
    @media (max-width:600px) { header { padding:15px 18px } main { width:min(100% - 28px,1180px); padding-top:36px } .grid { grid-template-columns:1fr } }
  </style>
</head>
<body>
  <header><div class="brand"><span class="mark"></span>SwapAI</div><div class="updated" id="updated">Connecting…</div></header>
  <main><h1>Classifiers</h1><p class="intro">Live training progress and measured accuracy from this SwapAI data directory.</p><div id="summary" class="summary"></div><div id="content"></div></main>
  <script>
    const content=document.querySelector('#content'),summary=document.querySelector('#summary'),updated=document.querySelector('#updated');
    const esc=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const pct=value=>value===null?'Not measured':new Intl.NumberFormat(undefined,{style:'percent',maximumFractionDigits:2}).format(value);
    const count=value=>new Intl.NumberFormat().format(value);
    const money=value=>value===null?'Not reported':'$'+Number(value).toFixed(3);
    const purposeCounts=purposes=>'train '+count(purposes.training||0)+' · validation '+count(purposes.validation||0)+' · representative '+count(purposes.representative_test||0)+' · coverage '+count(purposes.coverage_test||0);
    const list=(title,items,empty='None')=>'<section class="fact"><h3>'+title+'</h3><ul>'+(items.length?items.join(''):'<li class="meta">'+empty+'</li>')+'</ul></section>';
    const renderRun=run=>{
      const evaluations=(run.evaluations||[]).map(metric=>'<li><strong>'+esc(metric.purpose)+'</strong> · '+esc(metric.resultBin||'all bins')+' · '+count(metric.exampleCount)+' examples · '+pct(metric.error)+' <span class="'+(metric.passed?'good':'bad')+'">'+(metric.passed?'pass':'fail')+'</span></li>');
      const resources=(run.resources||[]).map(resource=>'<li>'+esc(resource.type)+' · '+esc(resource.id)+'</li>');
      const shadow=run.shadow===null?'<li class="meta">No live shadow evidence yet</li>':'<li>'+count(run.shadow.exampleCount)+' examples · '+pct(run.shadow.meanError)+' mean error · <span class="'+(run.shadow.passed?'good':'bad')+'">'+(run.shadow.passed?'pass':'fail')+'</span> · '+count(run.shadow.failureCount)+' failures'+(run.shadow.lastFailureMessage?'<div class="failure">'+esc(run.shadow.lastFailureMessage)+'</div>':'')+'</li>';
      return '<div class="run"><div class="run-head"><strong>'+esc(run.provider)+' · '+esc(run.status)+'</strong><span class="meta">'+esc(run.id)+'</span></div><div class="meta">Dataset '+esc(run.datasetRevisionId)+' · Provider run '+esc(run.providerRunId||'not assigned')+' · Cost '+money(run.costUsd)+'</div>'+(run.failureMessage?'<div class="failure">'+esc(run.failureMessage)+'</div>':'')+list('Protected evaluation',evaluations)+list('Shadow evidence',[shadow])+list('Provider resources',resources)+list('Cleanup',['<li class="'+(run.cleanup.status==='failed'?'failure':'')+'">'+esc(run.cleanup.status)+(run.cleanup.message?' · '+esc(run.cleanup.message):'')+'</li>'])+'</div>';
    };
    const renderClassifier=item=>{
      const measured=item.lastEvaluatedError,within=measured!==null&&measured<=item.acceptableError,last=item.latestTrainingRun,purposes=item.examplesByPurpose||{},covered=item.resultBins.filter(bin=>bin.total>0).length;
      const deficits=(item.deficits||[]).map(gap=>'<li>'+esc(gap.purpose)+' · '+esc(gap.resultBin||'all bins')+' · '+count(gap.available)+' / '+count(gap.required)+'</li>');
      const bins=(item.resultBins||[]).map(bin=>'<li><strong>'+esc(bin.id)+'</strong><div class="meta">'+purposeCounts(bin.purposes||{})+'</div></li>');
      const facets=(item.facetCoverage||[]).map(facet=>'<li><strong>'+esc(facet.facet)+' = '+esc(facet.value===null?'(unlabelled)':facet.value)+'</strong><div class="meta">'+purposeCounts(facet.purposes||{})+'</div></li>');
      const runs=(item.trainingRuns||[]).map(renderRun);
      return '<article class="card"><div class="card-head"><div><h2>'+esc(item.name)+'</h2><div class="type">'+esc(item.resultType)+' result · '+esc(item.needleVersion||'No promoted model')+'</div></div><div class="states"><span class="pill '+(item.readyForTraining?'on':'')+'">'+(item.readyForTraining?'Dataset ready':count(item.deficits.length)+' gaps')+'</span><span class="pill '+(item.training?'training':'')+'">'+(item.training?'Training':'Idle')+'</span><span class="pill '+(item.trained?'on':'')+'">'+(item.trained?'Promoted':'Reference active')+'</span></div></div><dl><dt>Retained / observed</dt><dd>'+count(item.retainedExamples)+' / '+count(item.totalExamplesLogged)+'</dd><dt>Covered result bins</dt><dd>'+count(covered)+' / '+count(item.resultBins.length)+'</dd><dt>Train / validate</dt><dd>'+count(purposes.training||0)+' / '+count(purposes.validation||0)+'</dd><dt>Representative / coverage test</dt><dd>'+count(purposes.representative_test||0)+' / '+count(purposes.coverage_test||0)+'</dd><dt>Worst protected-test error</dt><dd class="'+(within?'good':measured===null?'':'bad')+'">'+pct(measured)+'</dd><dt>Required error</dt><dd>'+pct(item.acceptableError)+' or less</dd><dt>Latest trainer</dt><dd>'+(last?esc(last.provider+' · '+last.status):'Never run')+'</dd><dt>Latest run cost</dt><dd>'+(last?money(last.costUsd):'Not reported')+'</dd></dl><div class="facts">'+list('Training deficits',deficits,'Dataset is ready')+list('Result-bin coverage',bins)+list('Facet coverage',facets,'No facets declared')+'<section class="fact"><h3>Training run history</h3>'+(runs.length?runs.join(''):'<div class="meta">Never run</div>')+'</section></div></article>';
    };
    async function refresh(){try{const response=await fetch('/api/classifiers',{cache:'no-store'});if(!response.ok)throw new Error('Status API returned '+response.status);const data=await response.json(),items=data.classifiers;updated.textContent='Updated '+new Date(data.generatedAt).toLocaleTimeString();summary.innerHTML='<span>'+count(items.length)+' classifiers</span><span>'+count(items.filter(x=>x.readyForTraining).length)+' ready</span><span>'+count(items.filter(x=>x.trained).length)+' promoted</span><span>'+count(items.filter(x=>x.training).length)+' training</span>';if(items.length===0){content.innerHTML='<div class="empty">No classifiers have used this data directory yet.</div>';return}content.className='grid';content.innerHTML=items.map(renderClassifier).join('')}catch(error){updated.textContent='Disconnected';content.className='';content.innerHTML='<div class="error">'+esc(error.message||error)+'</div>'}}
    refresh();setInterval(refresh,3000);
  </script>
</body>
</html>`;
