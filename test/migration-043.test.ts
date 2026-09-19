import { createRequire } from "node:module";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createClassifier } from "../src/index.js";

const { DatabaseSync } = createRequire(import.meta.url)(
  "node:sqlite",
) as typeof import("node:sqlite");
const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("0.4.3 storage migration", () => {
  it("preserves training evidence and foreign keys across repeated opens", async () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "swapai-043-migration-"));
    directories.push(dataDirectory);
    create043Database(join(dataDirectory, "swapai.sqlite"));

    for (let open = 0; open < 2; open += 1) {
      const classifier = createClassifier({
        name: "migration",
        result: { type: "boolean" },
        reference: async () => true,
        dataDirectory,
      });
      await classifier.close();
    }

    const database = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    database.function("swapai_writer_version", { deterministic: true }, () => 3);
    expect(database.prepare("PRAGMA foreign_key_check").all()).toEqual([]);
    expect(database.prepare("SELECT COUNT(*) AS count FROM training_runs").get())
      .toEqual({ count: 1 });
    expect(database.prepare("SELECT COUNT(*) AS count FROM training_evaluations").get())
      .toEqual({ count: 1 });
    expect(database.prepare("SELECT COUNT(*) AS count FROM model_artifacts").get())
      .toEqual({ count: 1 });
    database.close();

    const oldWriter = new DatabaseSync(join(dataDirectory, "swapai.sqlite"));
    oldWriter.function("swapai_writer_version", { deterministic: true }, () => 2);
    expect(() => oldWriter.prepare(`
      UPDATE classifiers SET updated_at = updated_at + 1 WHERE name = 'migration'
    `).run()).toThrow(/writer is too old/i);
    oldWriter.close();
  });
});

const create043Database = (path: string): void => {
  const database = new DatabaseSync(path);
  database.exec(`
    PRAGMA foreign_keys = ON;
    CREATE TABLE classifiers (
      name TEXT PRIMARY KEY, config_json TEXT NOT NULL, max_training_set INTEGER NOT NULL,
      active_generation INTEGER NOT NULL DEFAULT 1, total_examples_logged INTEGER NOT NULL DEFAULT 0,
      new_examples_since_training INTEGER NOT NULL DEFAULT 0, training_attempts INTEGER NOT NULL DEFAULT 0,
      examples_used_for_training INTEGER NOT NULL DEFAULT 0, last_evaluated_error REAL,
      last_evaluated_at INTEGER, local_classifications_since_retest INTEGER NOT NULL DEFAULT 0,
      total_local_classifications INTEGER NOT NULL DEFAULT 0, total_retests INTEGER NOT NULL DEFAULT 0,
      consecutive_retest_failures INTEGER NOT NULL DEFAULT 0, training_lease_owner TEXT,
      training_lease_until INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
      data_epoch INTEGER NOT NULL DEFAULT 0, clear_pending INTEGER NOT NULL DEFAULT 0,
      clear_erased INTEGER NOT NULL DEFAULT 0, clear_artifact_generation_max INTEGER,
      training_lease_epoch INTEGER
    );
    CREATE TABLE generations (
      classifier_name TEXT NOT NULL, generation INTEGER NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('active','archived')), trained INTEGER NOT NULL DEFAULT 0,
      model_path TEXT, needle_version TEXT, created_at INTEGER NOT NULL, archived_at INTEGER,
      PRIMARY KEY(classifier_name,generation),
      FOREIGN KEY(classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
    );
    CREATE TABLE examples (
      id INTEGER PRIMARY KEY AUTOINCREMENT, classifier_name TEXT NOT NULL, generation INTEGER NOT NULL,
      input TEXT NOT NULL, result_json TEXT NOT NULL,
      split TEXT NOT NULL CHECK(split IN ('training','held_out')), input_hash TEXT NOT NULL DEFAULT '',
      result_bin TEXT NOT NULL DEFAULT 'legacy', purpose TEXT NOT NULL DEFAULT 'legacy_seen',
      facets_json TEXT NOT NULL DEFAULT '{}', created_at INTEGER NOT NULL,
      FOREIGN KEY(classifier_name,generation) REFERENCES generations(classifier_name,generation) ON DELETE CASCADE
    );
    CREATE TABLE classifier_runtimes (
      classifier_name TEXT NOT NULL, runtime_id TEXT NOT NULL, pid INTEGER NOT NULL,
      started_at INTEGER NOT NULL, heartbeat_at INTEGER NOT NULL,
      PRIMARY KEY(classifier_name,runtime_id),
      FOREIGN KEY(classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
    );
    CREATE TABLE dataset_revisions (
      id TEXT PRIMARY KEY, classifier_name TEXT NOT NULL, generation INTEGER NOT NULL,
      data_epoch INTEGER NOT NULL, created_at INTEGER NOT NULL,
      FOREIGN KEY(classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
    );
    CREATE TABLE dataset_revision_examples (
      dataset_revision_id TEXT NOT NULL, position INTEGER NOT NULL, input TEXT NOT NULL,
      result_json TEXT NOT NULL, result_bin TEXT NOT NULL,
      purpose TEXT NOT NULL CHECK(purpose IN ('training','validation','representative_test','coverage_test')),
      facets_json TEXT NOT NULL, input_hash TEXT NOT NULL,
      PRIMARY KEY(dataset_revision_id,position),
      FOREIGN KEY(dataset_revision_id) REFERENCES dataset_revisions(id) ON DELETE CASCADE
    );
    CREATE TABLE training_runs (
      id TEXT PRIMARY KEY, dataset_revision_id TEXT NOT NULL, classifier_name TEXT NOT NULL,
      provider_name TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('running','failed','rejected','promoted')),
      provider_run_id TEXT, cost_usd REAL, artifact_sha256 TEXT, failure_message TEXT,
      started_at INTEGER NOT NULL, finished_at INTEGER,
      FOREIGN KEY(dataset_revision_id) REFERENCES dataset_revisions(id) ON DELETE CASCADE,
      FOREIGN KEY(classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
    );
    CREATE TABLE training_evaluations (
      training_run_id TEXT NOT NULL,
      purpose TEXT NOT NULL CHECK(purpose IN ('representative_test','coverage_test')),
      result_bin TEXT, example_count INTEGER NOT NULL, error REAL NOT NULL, passed INTEGER NOT NULL,
      PRIMARY KEY(training_run_id,purpose,result_bin),
      FOREIGN KEY(training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
    );
    CREATE TABLE model_artifacts (
      classifier_name TEXT NOT NULL, sha256 TEXT NOT NULL, model_path TEXT NOT NULL,
      needle_version TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL,
      PRIMARY KEY(classifier_name,sha256),
      FOREIGN KEY(classifier_name) REFERENCES classifiers(name) ON DELETE CASCADE
    );
  `);
  const now = Date.now();
  const config = JSON.stringify({
    result: { type: "boolean" }, retrainOnCount: 50, acceptableError: 0.1,
    retestInterval: 100, retestRevertOn: 3, model: "needle2",
  });
  database.prepare("INSERT INTO classifiers(name,config_json,max_training_set,created_at,updated_at) VALUES('migration',?,10000,?,?)")
    .run(config, now, now);
  database.prepare("INSERT INTO generations(classifier_name,generation,status,created_at) VALUES('migration',1,'active',?)").run(now);
  database.prepare("INSERT INTO dataset_revisions(id,classifier_name,generation,data_epoch,created_at) VALUES('revision','migration',1,0,?)").run(now);
  database.prepare("INSERT INTO dataset_revision_examples VALUES('revision',0,'yes','true','true','representative_test','{}','hash')").run();
  database.prepare("INSERT INTO training_runs VALUES('run','revision','migration','local','rejected','provider',0.1,'artifact',NULL,?,?)").run(now, now);
  database.prepare("INSERT INTO training_evaluations VALUES('run','representative_test','true',1,0,1)").run();
  database.prepare("INSERT INTO model_artifacts VALUES('migration','artifact','/tmp/model.cact','2.0.14',1,?)").run(now);
  database.close();
};
