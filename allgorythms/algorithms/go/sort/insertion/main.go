package insertion

func InsertionSort(list []int) {
	for i := 1; i < len(list); i++ {
		j := i
		for j > 0 && list[j] < list[j-1] {
			list[j], list[j-1] = list[j-1], list[j]
			j--
		}
	}
}
