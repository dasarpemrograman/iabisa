import Box from "@mui/material/Box";

export function Grid({ children, columns, gap = 0 }) {
	return (
		<Box
			sx={{
				display: "grid",
				gap,
				// gridTemplateColumns: `repeat(${columns}, 1fr)`,
				gridAutoRows: 50,
				gridAutoFlow: "row dense",
				gridTemplateColumns: {
					xs: "repeat(4, 1fr)",
					sm: "repeat(4, 1fr)",
					md: "repeat(8, 1fr)",
					lg: "repeat(12, 1fr)",
					xl: "repeat(12, 1fr)",
				},
			}}
		>
			{children}
		</Box>
	);
}
