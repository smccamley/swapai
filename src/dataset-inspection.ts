import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

import { validateResult } from "./config.js";
import {
  prepareExampleMetadata,
  resultBins,
} from "./dataset-policy.js";
import type {
  ClassifierInspection,
  DatasetDeficit,
  DatasetPolicy,
  DatasetPurpose,
  ResultBinInspection,
  ResultConfig,
  TrainingRunInspection,
} from "./types.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");

interface CountRow {
  result_bin: string;
  purpose: DatasetPurpose;
  example_count: number;
}

interface LegacyExampleRow {
  input: string;
  result_json: string;
  split: "training" | "held_out";
}

const emptyPurposes = () => ({
  training: 0,
  validation: 0,
  representative_test: 0,
  coverage_test: 0,
});

export const readClassifierInspection = (options: {
  readonly dataDirectory: string;
  readonly name: string;
  readonly result: ResultConfig;
  readonly acceptableError: number;
  readonly policy: DatasetPolicy;
}): ClassifierInspection => {
  const databasePath = join(options.dataDirectory, "swapai.sqlite");
  const bins = resultBins(
    options.result,
    options.acceptableError,
    options.policy,
  );
  if (!existsSync(databasePath)) {
    return inspectionFromCounts(options.name, bins, options.policy, 0, []);
  }

  const database = new DatabaseSync(databasePath, { readOnly: true });
  try {
    database.exec("PRAGMA busy_timeout = 5000");
    const classifier = database.prepare(`
      SELECT active_generation, total_examples_logged
      FROM classifiers
      WHERE name = ?
    `).get(options.name) as
      | { active_generation: number; total_examples_logged: number }
      | undefined;
    if (classifier === undefined) {
      return inspectionFromCounts(options.name, bins, options.policy, 0, []);
    }
    const exampleColumns = tableColumns(database, "examples");
    const counts = exampleColumns.has("result_bin") && exampleColumns.has("purpose")
      ? database.prepare(`
          SELECT result_bin, purpose, COUNT(*) AS example_count
          FROM examples
          WHERE classifier_name = ?
            AND generation = ?
            AND purpose != 'legacy_seen'
          GROUP BY result_bin, purpose
        `).all(options.name, classifier.active_generation) as unknown as CountRow[]
      : legacyExampleCounts(database, options, classifier.active_generation);
    const latestTrainingRun = hasTable(database, "training_runs")
      ? mapTrainingRun(database.prepare(`
          SELECT id, dataset_revision_id, provider_name, status,
                 provider_run_id, cost_usd, artifact_sha256, failure_message,
                 started_at, finished_at
          FROM training_runs
          WHERE classifier_name = ?
          ORDER BY started_at DESC, id DESC
          LIMIT 1
        `).get(options.name))
      : null;
    return inspectionFromCounts(
      options.name,
      bins,
      options.policy,
      classifier.total_examples_logged,
      counts,
      latestTrainingRun,
    );
  } finally {
    database.close();
  }
};

const inspectionFromCounts = (
  name: string,
  bins: ReturnType<typeof resultBins>,
  policy: DatasetPolicy,
  totalExamplesLogged: number,
  counts: readonly CountRow[],
  latestTrainingRun: TrainingRunInspection | null = null,
): ClassifierInspection => {
  const byBin = new Map<string, ReturnType<typeof emptyPurposes>>();
  for (const bin of bins) byBin.set(bin.id, emptyPurposes());
  for (const count of counts) {
    if (count.purpose === "legacy_seen") continue;
    const purposes = byBin.get(count.result_bin);
    if (purposes !== undefined) purposes[count.purpose] = count.example_count;
  }

  const resultBinInspections: ResultBinInspection[] = bins.map((bin) => {
    const purposes = byBin.get(bin.id) ?? emptyPurposes();
    const total = Object.values(purposes).reduce(
      (sum, count) => sum + count,
      0,
    );
    return "minimum" in bin
      ? {
          id: bin.id,
          range: {
            minimum: bin.minimum,
            maximum: bin.maximum,
            includesMaximum: bin.includesMaximum,
          },
          total,
          purposes,
        }
      : { id: bin.id, value: bin.value, total, purposes };
  });

  const deficits: DatasetDeficit[] = [];
  const trainingTotal = resultBinInspections.reduce(
    (sum, bin) => sum + bin.purposes.training,
    0,
  );
  const representativeTotal = resultBinInspections.reduce(
    (sum, bin) => sum + bin.purposes.representative_test,
    0,
  );
  addDeficit(
    deficits,
    "training",
    null,
    policy.requirements.minimumTrainingExamples,
    trainingTotal,
  );
  addDeficit(
    deficits,
    "representative_test",
    null,
    policy.requirements.minimumRepresentativeTestExamples,
    representativeTotal,
  );
  for (const bin of resultBinInspections) {
    addDeficit(
      deficits,
      "training",
      bin.id,
      policy.requirements.minimumTrainingExamplesPerResultBin,
      bin.purposes.training,
    );
    addDeficit(
      deficits,
      "validation",
      bin.id,
      policy.requirements.minimumValidationExamplesPerResultBin,
      bin.purposes.validation,
    );
    addDeficit(
      deficits,
      "coverage_test",
      bin.id,
      policy.requirements.minimumCoverageTestExamplesPerResultBin,
      bin.purposes.coverage_test,
    );
  }

  return {
    name,
    totalExamplesLogged,
    retainedExamples: resultBinInspections.reduce(
      (sum, bin) => sum + bin.total,
      0,
    ),
    readyForTraining: deficits.length === 0,
    resultBins: resultBinInspections,
    deficits,
    examplesByPurpose: resultBinInspections.reduce(
      (totals, bin) => ({
        training: totals.training + bin.purposes.training,
        validation: totals.validation + bin.purposes.validation,
        representative_test:
          totals.representative_test + bin.purposes.representative_test,
        coverage_test: totals.coverage_test + bin.purposes.coverage_test,
      }),
      emptyPurposes(),
    ),
    latestTrainingRun,
  };
};

const hasTable = (
  database: InstanceType<typeof DatabaseSync>,
  name: string,
): boolean => database.prepare(`
  SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?
`).get(name) !== undefined;

const tableColumns = (
  database: InstanceType<typeof DatabaseSync>,
  table: string,
): Set<string> => new Set(
  database.prepare(`PRAGMA table_info(${table})`).all().map(
    (value) => (value as { name: string }).name,
  ),
);

const legacyExampleCounts = (
  database: InstanceType<typeof DatabaseSync>,
  options: Parameters<typeof readClassifierInspection>[0],
  generation: number,
): CountRow[] => {
  const examples = database.prepare(`
    SELECT input, result_json, split
    FROM examples
    WHERE classifier_name = ? AND generation = ?
    ORDER BY id
  `).all(options.name, generation) as unknown as LegacyExampleRow[];
  const counts = new Map<string, CountRow>();
  for (const example of examples) {
    const result = validateResult(options.result, JSON.parse(example.result_json));
    const metadata = prepareExampleMetadata(
      options.name,
      options.result,
      options.acceptableError,
      options.policy,
      example.input,
      result,
    );
    const purpose = example.split === "training" ? "training" : "validation";
    const key = `${metadata.resultBin}\0${purpose}`;
    const existing = counts.get(key);
    counts.set(key, {
      result_bin: metadata.resultBin,
      purpose,
      example_count: (existing?.example_count ?? 0) + 1,
    });
  }
  return [...counts.values()];
};

const mapTrainingRun = (value: unknown): TrainingRunInspection | null => {
  if (value === undefined) return null;
  const row = value as {
    id: string;
    dataset_revision_id: string;
    provider_name: string;
    status: TrainingRunInspection["status"];
    provider_run_id: string | null;
    cost_usd: number | null;
    artifact_sha256: string | null;
    failure_message: string | null;
    started_at: number;
    finished_at: number | null;
  };
  return {
    id: row.id,
    datasetRevisionId: row.dataset_revision_id,
    provider: row.provider_name,
    status: row.status,
    providerRunId: row.provider_run_id,
    costUsd: row.cost_usd,
    artifactSha256: row.artifact_sha256,
    failureMessage: row.failure_message,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
  };
};

const addDeficit = (
  deficits: DatasetDeficit[],
  purpose: DatasetDeficit["purpose"],
  resultBin: string | null,
  required: number,
  available: number,
): void => {
  if (available >= required) return;
  deficits.push({ purpose, resultBin, required, available });
};
