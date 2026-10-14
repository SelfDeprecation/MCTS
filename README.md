# Параллелизация MCTS в AlphaZero

Исследовательская реализация последовательного поиска, классических Leaf / Root / Tree и Asynchronous Batched MCTS. Поддерживаются «Крестики-нолики» и «Четыре в ряд», самостоятельная игра, обучение сети политики и значения, сравнение оптимизаторов AdamW / ASAM / Muon с обрезанием градиента и воспроизводимые измерения.

Это компактная учебная реализация принципов AlphaZero с полносвязной остаточной сетью, а не воспроизведение большой шахматной системы DeepMind. Точный минимаксный решатель используется только для проверки малой игры, не для обучения.

## Установка и проверка

Требуются Python 3.12 и `uv`. Из корня репозитория:

```powershell
uv sync --python 3.12 --extra dev
uv run az-study doctor
uv run pytest -q
```

В Windows устанавливается CPU-сборка PyTorch 2.9.1. Для запуска GPU используйте ноутбук Google Colab.

## Обучение и эксперименты на CPU

```powershell
uv run az-study train --config configs/train_cpu.json --out runs/train_cpu
uv run az-study benchmark --config configs/benchmark_cpu.json --checkpoint runs/train_cpu/latest.pt --out runs/benchmark_cpu
uv run az-study benchmark --config configs/benchmark_rollout.json --out runs/rollout_cpu
uv run az-study benchmark --config configs/benchmark_time.json --checkpoint runs/train_cpu/latest.pt --out runs/time_cpu
uv run az-study optimizers --config configs/optimizers_cpu.json --replay runs/train_cpu/latest.pt --out runs/optimizers_cpu
uv run az-study arena --checkpoint runs/train_cpu/latest.pt --opponent minimax --games 100 --out runs/arena_minimax.json
uv run az-study play --checkpoint runs/train_cpu/latest.pt
```

Команды не перезаписывают завершённые эксперименты: для повторения задавайте новый `--out`. Продолжение обучения:

```powershell
uv run az-study train --config configs/train_cpu.json --out runs/train_cpu --resume runs/train_cpu/latest.pt
```

Контрольная точка сохраняет сеть, оптимизатор, буфер, номер итерации и состояния генераторов. Порядок завершения запросов многопоточного поиска зависит от планировщика; побитовая воспроизводимость для него не гарантируется.

## Google Colab

Создайте архив исходников:

```powershell
uv run python scripts/package_colab.py --out dist/parallel-alphazero-colab.zip
```

Откройте [ноутбук](notebooks/colab.ipynb) в Colab, выберите среду с GPU и выполните ячейки по порядку. Ноутбук загружает архив, проверяет CUDA, запускает тесты и записывает результаты на Google Drive.

В Colab не выполняйте `uv sync` с локальным Windows-окружением. Ноутбук устанавливает проект с `pip install --no-deps -e .`, сохраняя CUDA-сборку PyTorch среды. Muon требует PyTorch не ниже 2.9.

Основная GPU-конфигурация: 50 итераций × 24 партии, 64 симуляции на ход, сеть шириной 256 с четырьмя остаточными блоками. Сравнение поиска: 256 симуляций, 16 позиций, три начальных числа генератора и три повтора; обученная контрольная точка передаётся явно. Длительность зависит от предоставленного оборудования.

## Структура

```text
src/azparallel/    игровые правила, сеть, поиск, обучение и измерения
configs/          конфигурации CPU и GPU
tests/            автоматические проверки
scripts/          запуск серий, анализ и упаковка Colab
notebooks/        ноутбук Google Colab
.github/workflows/ проверки Windows и Linux
pyproject.toml    зависимости и команда az-study
uv.lock           зафиксированные версии зависимостей
```

## Ограничения сравнения

При анализе учитываются не только возвраты, но и фактические оценки листьев, расширенные узлы и качество действий. Нейросетевой Leaf усредняет допустимые симметрии одного листа; классический Leaf использует случайные доигрывания. У Root бюджет разделяется между деревьями, у Tree и Async резервы отделены от завершённой статистики.

Tree использует глобальную блокировку коротких операций дерева. Нейросетевые схемы используют потоки и один защищённый оценщик; Async объединяет запросы в пакеты. Это асинхронность работников поиска и сервера оценок, а не ручное перекрытие нескольких CUDA streams. Для классических Leaf и Root доступны процессы с режимом `spawn`.

Время поиска включает подготовку корня и завершение принятых запросов; прогрев исключён. Бутстрэп выполняется по позициям, не по отдельным повторам. Сравнивать CPU и GPU как аппаратное ускорение можно только при одинаковой игре, сети, весах и протоколе. Победы против случайного соперника не доказывают оптимальную игру.
