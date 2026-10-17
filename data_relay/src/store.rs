//! File-backed task store: one directory + `task.json` per task, atomic writes.

use crate::model::{Task, TaskStatus};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct TaskStore {
    tasks_dir: PathBuf,
    inner: Mutex<BTreeMap<String, Task>>,
}

impl TaskStore {
    pub fn open(tasks_dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(tasks_dir)?;
        let mut map = BTreeMap::new();
        for entry in std::fs::read_dir(tasks_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let manifest = entry.path().join("task.json");
            if !manifest.is_file() {
                continue;
            }
            match serde_json::from_str::<Task>(&std::fs::read_to_string(&manifest)?) {
                Ok(task) => {
                    map.insert(task.id.clone(), task);
                }
                Err(e) => tracing::warn!("跳过损坏的任务清单 {}: {e}", manifest.display()),
            }
        }
        Ok(TaskStore {
            tasks_dir: tasks_dir.to_path_buf(),
            inner: Mutex::new(map),
        })
    }

    pub fn list(&self) -> Vec<Task> {
        self.inner.lock().unwrap().values().rev().cloned().collect()
    }

    pub fn get(&self, id: &str) -> Option<Task> {
        self.inner.lock().unwrap().get(id).cloned()
    }

    pub fn task_dir(&self, id: &str) -> PathBuf {
        self.tasks_dir.join(id)
    }

    pub fn insert(&self, task: Task) -> anyhow::Result<()> {
        let dir = self.task_dir(&task.id);
        std::fs::create_dir_all(&dir)?;
        self.save(&task)?;
        self.inner.lock().unwrap().insert(task.id.clone(), task);
        Ok(())
    }

    /// Mutate a task and persist. Returns the updated copy.
    pub fn update<F: FnOnce(&mut Task)>(&self, id: &str, f: F) -> anyhow::Result<Task> {
        let mut map = self.inner.lock().unwrap();
        let task = map
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("任务 {id} 不存在"))?;
        f(task);
        self.save(task)?;
        Ok(task.clone())
    }

    pub fn update_status(&self, id: &str, status: TaskStatus) -> anyhow::Result<Task> {
        self.update(id, |t| {
            t.status = status;
            match status {
                TaskStatus::Running => t.started_at = Some(crate::model::now_secs()),
                TaskStatus::Done | TaskStatus::Failed | TaskStatus::Cancelled => {
                    t.finished_at = Some(crate::model::now_secs())
                }
                _ => {}
            }
        })
    }

    pub fn remove(&self, id: &str) -> anyhow::Result<()> {
        {
            let map = self.inner.lock().unwrap();
            let task = map
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("任务 {id} 不存在"))?;
            if matches!(task.status, TaskStatus::Running | TaskStatus::Uploading) {
                anyhow::bail!("任务正在运行，请先取消再删除");
            }
        }
        self.inner.lock().unwrap().remove(id);
        let dir = self.task_dir(id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    fn save(&self, task: &Task) -> anyhow::Result<()> {
        let path = self.task_dir(&task.id).join("task.json");
        let tmp = self.task_dir(&task.id).join("task.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(task)?)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{UploadMode, now_secs};

    fn sample(id: &str) -> Task {
        Task {
            id: id.into(),
            name: "demo".into(),
            status: TaskStatus::Queued,
            created_at: now_secs(),
            started_at: None,
            finished_at: None,
            script: "run.py".into(),
            interpreter: "python3".into(),
            output_dir: "out".into(),
            exit_code: None,
            auto_upload: true,
            upload_mode: UploadMode::Sftp,
            remote_path: None,
            upload_total_bytes: 0,
            upload_done_bytes: 0,
            pull_token: None,
            error: None,
        }
    }

    #[test]
    fn insert_update_persist_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("relay_test_{}", uuid::Uuid::new_v4()));
        let store = TaskStore::open(&tmp).unwrap();
        store.insert(sample("t1")).unwrap();
        assert_eq!(store.list().len(), 1);
        store.update("t1", |t| t.error = Some("boom".into())).unwrap();
        // reopen from disk: persistence check
        let store2 = TaskStore::open(&tmp).unwrap();
        assert_eq!(store2.get("t1").unwrap().error.as_deref(), Some("boom"));
        store2.remove("t1").unwrap();
        assert!(store2.get("t1").is_none());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn cannot_remove_running_task() {
        let tmp = std::env::temp_dir().join(format!("relay_test_{}", uuid::Uuid::new_v4()));
        let store = TaskStore::open(&tmp).unwrap();
        store.insert(sample("t2")).unwrap();
        store.update_status("t2", TaskStatus::Running).unwrap();
        assert!(store.remove("t2").is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }
}
