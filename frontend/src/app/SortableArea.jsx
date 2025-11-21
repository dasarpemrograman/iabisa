import { useSortable } from "@dnd-kit/sortable";

import { Area } from "./Area";

export const SortableArea = ({ id, ...props }) => {
	const {
		attributes,
		isDragging,
		listeners,
		setNodeRef,
		transform,
		transition,
	} = useSortable({ id });

	return (
		<Area
			ref={setNodeRef}
			id={id}
			{...props}
			style={{
				"--translate-x": transform ? `${transform.x}px` : 0,
				"--translate-y": transform ? `${transform.y}px` : 0,
				"--transition": transition ?? "none",
			}}
			placeholder={isDragging}
			dragHandle={{ ...attributes, ...listeners }}
		/>
	);
};
