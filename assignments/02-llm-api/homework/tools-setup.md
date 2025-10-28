# Установка инструментов

## 📋 Список необходимых инструментов

- ✅ **uv** - пакетный менеджер Python
- ✅ **Python 3.11+**
- ✅ **Git**
- ✅ **make**

### Опционально:
- **Warp Terminal** - для упрощения установки

---

## 🚀 Быстрая установка

### Основные инструменты

#### Через Warp Terminal (рекомендуется)

Установите [Warp](https://www.warp.dev/), затем используйте Agent Mode (✨):

```bash
✨ установи uv пакетный менеджер python
✨ установи python 3.11
```

#### Ручная установка

**uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Python через uv:**
```bash
uv python install 3.11
```

**Git:**
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git

# Windows - скачайте с git-scm.com
```

**make:**
```bash
# macOS
brew install make

# Ubuntu/Debian
sudo apt-get install build-essential

# Windows
choco install make
# или используйте Git Bash (make уже включен)
```

---

## ✅ Проверка установки

Выполните команды для проверки:

```bash
# Основные инструменты
uv --version          # должно быть 0.4+
python --version      # должно быть 3.11+
git --version
make --version
```

---

## 🔧 Конфигурация Git

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@systtech.ru"
git config --global init.defaultBranch main
```

