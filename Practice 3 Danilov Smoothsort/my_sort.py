"""Модуль my_sort: реализация Smoothsort с визуализацией леса куч Леонардо.

Предоставляет функции для плавной сортировки с пошаговым отображением
обменов, состояния массива и структуры куч Леонардо.
"""

import random
import sys
from typing import Any, Callable, List, Optional


def get_leonardo_numbers(max_size: int) -> List[int]:
    """Вычисляет последовательность чисел Леонардо до заданного размера."""
    leo = list((1, 1))
    while leo[-1] <= max_size:
        leo.append(leo[-1] + leo[-2] + 1)
    return leo


def _cmp_to_key(cmp_func: Callable[[Any, Any], int]) -> Callable:
    """Преобразует функцию сравнения (cmp) в функцию-ключ для сортировки."""

    class KeyWrapper:
        """Класс-обёртка для поддержки cmp-сравнений на всех этапах."""

        __slots__ = ("obj",)

        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def __lt__(self, other: "KeyWrapper") -> bool:
            return cmp_func(self.obj, other.obj) < 0

        def __gt__(self, other: "KeyWrapper") -> bool:
            return cmp_func(self.obj, other.obj) > 0

        def __le__(self, other: "KeyWrapper") -> bool:
            return cmp_func(self.obj, other.obj) <= 0

        def __ge__(self, other: "KeyWrapper") -> bool:
            return cmp_func(self.obj, other.obj) >= 0

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, KeyWrapper):
                return NotImplemented
            return cmp_func(self.obj, other.obj) == 0

        def __ne__(self, other: object) -> bool:
            if not isinstance(other, KeyWrapper):
                return NotImplemented
            return cmp_func(self.obj, other.obj) != 0

    return KeyWrapper


class StepVisualizer:
    """Утилита для пошаговой визуализации процесса сортировки."""

    def __init__(self) -> None:
        """Инициализирует визуализатор, устанавливая счётчик шагов в ноль."""
        self.step_count = 0

    def on_swap(self, i: int, j: int) -> None:
        """Вызывается при каждом обмене элементов массива."""
        self.step_count += 1
        print(f"Шаг {self.step_count}: обмен индексов {i} <-> {j}")

    @staticmethod
    def show_array(arr: List[Any], note: str = "") -> None:
        """Выводит текущее состояние массива с пояснением."""
        arr_vals = [str(x) for x in arr]
        prefix = f"{note}: " if note else ""
        print(f"{prefix}Массив: [{', '.join(arr_vals)}]")


def _default_key(x: Any) -> Any:
    """Функция-ключ по умолчанию."""
    return x


# pylint: disable=too-few-public-methods
class _SmoothSorter:
    """Внутренний класс для выполнения алгоритма Smoothsort."""

    def __init__(
        self,
        arr: List[Any],
        reverse: bool,
        key_func: Callable,
        callbacks: tuple,
    ) -> None:
        """Инициализирует состояние сортировщика."""
        self.lst = list(arr)
        self.size = len(arr)
        self.reverse = reverse
        self.key_func = key_func
        self.callbacks = callbacks
        self.leo = get_leonardo_numbers(self.size)
        self.state = {
            "p_mask": 1,
            "b_order": 1,
            "active_len": 0,
            "is_phase_two": False,
        }

    def _format_forest(self) -> str:
        """Формирует строковое представление текущего леса куч Леонардо."""
        active_len = self.state["active_len"]
        mask = self.state["p_mask"]
        if active_len <= 0 or mask == 0:
            return f"Готово: {self.lst}"

        orders = []
        order = self.state["b_order"]
        while mask > 0:
            if mask & 1:
                orders.append(order)
            mask >>= 1
            order += 1

        orders.reverse()
        heaps = []
        idx = 0
        for ord_val in orders:
            h_size = self.leo[ord_val]
            end_idx = idx + h_size
            chunk = self.lst[idx:end_idx]
            heaps.append(f"L({ord_val}): {chunk}")
            idx = end_idx

        forest_str = " | ".join(heaps)
        if active_len < self.size:
            tail = self.lst[active_len:]
            label = "Отсорт" if self.state["is_phase_two"] else "Необраб"
            return f"Лес: {forest_str} | {label}: {tail}"
        return f"Лес: {forest_str}"

    def _notify_state(self, context: str) -> None:
        """Передаёт текущее состояние массива и леса куч в визуализатор."""
        _, on_array_state = self.callbacks
        if on_array_state:
            full_note = f"{context} [{self._format_forest()}]"
            on_array_state(self.lst, full_note)

    def _is_greater(self, idx1: int, idx2: int) -> bool:
        """Проверяет, должен ли элемент idx1 находиться выше idx2 в куче."""
        val1 = self.key_func(self.lst[idx1])
        val2 = self.key_func(self.lst[idx2])
        return val1 < val2 if self.reverse else val1 > val2

    def _swap(self, i: int, j: int, context: str) -> None:
        """Меняет элементы местами и уведомляет визуализатор."""
        self.lst[i], self.lst[j] = self.lst[j], self.lst[i]
        on_swap, _ = self.callbacks
        if on_swap:
            on_swap(i, j)
        self._notify_state(context)

    def _sift(self, root: int, order: int) -> None:
        """Просеивает элемент вниз внутри кучи Леонардо заданного порядка."""
        curr = root
        curr_order = order
        while curr_order > 1:
            rt_child = curr - 1
            lf_child = curr - 1 - self.leo[curr_order - 2]

            if self._is_greater(lf_child, rt_child):
                biggest = lf_child
            else:
                biggest = rt_child

            if not self._is_greater(biggest, curr):
                break

            self._swap(curr, biggest, f"sift L({curr_order})")
            curr = biggest
            curr_order -= 1 if biggest == lf_child else 2

    def _trinkle(
        self,
        p_mask: int,
        order: int,
        root: int,
        trusty: bool,
    ) -> None:
        """Упорядочивает корни куч в лесу Леонардо."""
        curr_p = p_mask
        curr_order = order
        curr_root = root
        is_trusty = trusty

        while curr_p > 1:
            while curr_p & 1 == 0:
                curr_p >>= 1
                curr_order += 1

            prev_root = curr_root - self.leo[curr_order]
            if prev_root < 0 or not self._is_greater(prev_root, curr_root):
                break

            if not is_trusty and curr_order > 1:
                rt_child = curr_root - 1
                lf_child = curr_root - 1 - self.leo[curr_order - 2]
                if (
                    not self._is_greater(prev_root, rt_child)
                    or not self._is_greater(prev_root, lf_child)
                ):
                    break

            self._swap(curr_root, prev_root, "trinkle (корни)")
            curr_root = prev_root
            is_trusty = False
            curr_p >>= 1
            curr_order += 1

        if not is_trusty:
            self._sift(curr_root, curr_order)

    def sort(self) -> List[Any]:
        """Запускает построение куч и последующее извлечение элементов."""
        self._build_heaps()
        self._extract_elements()
        return self.lst

    def _build_heaps(self) -> None:
        """Фаза 1: построение леса куч Леонардо."""
        p_mask = 1
        b_order = 1
        self.state.update(
            {"p_mask": p_mask, "b_order": b_order, "active_len": 1}
        )
        self._notify_state("Фаза 1 (старт)")

        for i in range(1, self.size):
            if p_mask & 3 == 3:
                p_mask >>= 2
                b_order += 2
                p_mask += 1
            elif b_order == 1:
                p_mask <<= 1
                b_order = 0
                p_mask += 1
            else:
                p_mask <<= b_order - 1
                b_order = 1
                p_mask += 1

            self.state.update(
                {"p_mask": p_mask, "b_order": b_order, "active_len": i + 1}
            )
            self._notify_state(f"Фаза 1 (+элемент {self.lst[i]})")

            is_last_branch = (
                (b_order == 0 and i + 1 == self.size)
                or (
                    b_order > 0
                    and i + 1 + self.leo[b_order - 1] >= self.size
                )
            )
            if is_last_branch:
                self._trinkle(p_mask, b_order, i, False)
            else:
                self._sift(i, b_order)

        self._trinkle(p_mask, b_order, self.size - 1, False)

    def _extract_elements(self) -> None:
        """Фаза 2: извлечение элементов и разрушение куч."""
        self.state["is_phase_two"] = True
        p_mask = self.state["p_mask"]
        b_order = self.state["b_order"]

        for i in range(self.size - 1, 0, -1):
            if b_order <= 1:
                p_mask -= 1
                while p_mask > 0 and p_mask & 1 == 0:
                    p_mask >>= 1
                    b_order += 1
                self.state.update(
                    {"p_mask": p_mask, "b_order": b_order, "active_len": i}
                )
                self._notify_state(f"Фаза 2 (зафиксирован {self.lst[i]})")
            else:
                p_mask -= 1
                p_mask <<= 2
                p_mask += 3
                b_order -= 2
                self.state.update(
                    {"p_mask": p_mask, "b_order": b_order, "active_len": i}
                )
                msg = f"Фаза 2 (распад кучи, зафиксирован {self.lst[i]})"
                self._notify_state(msg)

                lf_root = i - 1 - self.leo[b_order]
                rt_root = i - 1
                self._trinkle(p_mask >> 1, b_order + 1, lf_root, True)
                self._trinkle(p_mask, b_order, rt_root, True)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def smoothsort(
    arr: List[Any],
    reverse: bool = False,
    key: Optional[Callable] = None,
    cmp: Optional[Callable[[Any, Any], int]] = None,
    on_swap: Optional[Callable[[int, int], None]] = None,
    on_array_state: Optional[Callable[[List[Any], str], None]] = None,
) -> List[Any]:
    """Реализация канонического алгоритма Smoothsort (Плавная сортировка)."""
    if len(arr) <= 1:
        return list(arr)

    if cmp is not None:
        key_func = _cmp_to_key(cmp)
    elif key is not None:
        key_func = key
    else:
        key_func = _default_key

    sorter = _SmoothSorter(arr, reverse, key_func, (on_swap, on_array_state))
    return sorter.sort()


def my_sort(
    array: List[Any],
    reverse: bool = False,
    key: Optional[Callable] = None,
    cmp: Optional[Callable[[Any, Any], int]] = None,
) -> List[Any]:
    """Публичная функция-обёртка для вызова плавной сортировки."""
    return smoothsort(array, reverse=reverse, key=key, cmp=cmp)


def read_numbers_from_file(path: str) -> List[Any]:
    """Читает данные из файла, автоматически определяя тип."""
    with open(path, "r", encoding="utf-8") as file_obj:
        content = file_obj.read().strip()
        if not content:
            return []
        tokens = content.split()
        try:
            return [int(x) for x in tokens]
        except ValueError:
            try:
                return [float(x) for x in tokens]
            except ValueError:
                return tokens


def write_data_to_file(path: str, data: List[Any]) -> None:
    """Записывает список данных в файл через пробел."""
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(" ".join(str(x) for x in data))


def _parse_tokens(tokens: List[str]) -> List[Any]:
    """Преобразует строковые токены в int, float или оставляет строками."""
    try:
        return [int(x) for x in tokens]
    except ValueError:
        try:
            return [float(x) for x in tokens]
        except ValueError:
            return tokens


def _run_interactive_mode(mode: str) -> None:
    """Запускает режим с клавиатурным вводом или случайной генерацией."""
    reverse_input = input(
        "Порядок (1=по возрастанию, 2=по убыванию): "
    ).strip()
    is_reverse = reverse_input == "2"
    visualizer = StepVisualizer()

    if mode == "1":
        raw = input("Введите элементы через пробел: ").strip()
        if not raw:
            print("Ничего не введено.")
            sys.exit(0)
        data_list = _parse_tokens(raw.split())
    else:
        try:
            prompt = "Введите размер случайного списка: "
            size_input = int(input(prompt).strip())
            data_list = [random.randint(1, 100) for _ in range(size_input)]
            print(f"Сгенерирован массив: {data_list}")
        except ValueError:
            print("Некорректный размер.")
            sys.exit(1)

    sorted_res = smoothsort(
        data_list,
        reverse=is_reverse,
        on_swap=visualizer.on_swap,
        on_array_state=visualizer.show_array,
    )
    print("\nИтоговый отсортированный массив:", sorted_res)
    print(f"Всего шагов (обменов): {visualizer.step_count}")


def _run_file_mode() -> None:
    """Запускает режим чтения из input.txt и записи в output.txt."""
    try:
        data_list = read_numbers_from_file("input.txt")
        if not data_list:
            print("Файл input.txt пуст.")
            sys.exit(0)
        print(f"Данные прочитаны ({len(data_list)} элементов).")
    except FileNotFoundError:
        print(
            "Файл input.txt не найден. Создайте его для выполнения сортировки."
        )
        sys.exit(1)

    reverse_input = input(
        "Порядок (1=по возрастанию, 2=по убыванию): "
    ).strip()
    is_reverse = reverse_input == "2"

    sorted_res = my_sort(data_list, reverse=is_reverse)
    write_data_to_file("output.txt", sorted_res)
    print("Готово: результат успешно записан в файл output.txt")


def main() -> None:
    """Точка входа для консольного интерфейса."""
    print("Выберите режим:")
    print("1 — Ввод с клавиатуры + подробная визуализация")
    print("2 — Чтение из input.txt, запись в output.txt")
    print("3 — Визуализация случайного списка заданного размера")
    mode = input("Режим (1, 2 или 3): ").strip()

    if mode in ("1", "3"):
        _run_interactive_mode(mode)
    elif mode == "2":
        _run_file_mode()
    else:
        print("Неверный режим.")


if __name__ == "__main__":
    main()
