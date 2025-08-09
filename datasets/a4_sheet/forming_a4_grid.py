# This file is to print A4 pattern on tiles to try to get height, and rotation
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4


def draw_tile_grid_on_a4(
    filename="tile_grid_10x5cm.pdf", tile_width_cm=3, tile_height_cm=5
):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    cols = int(width // (tile_width_cm * cm))
    rows = int(height // (tile_height_cm * cm))

    # Draw vertical lines
    for i in range(cols + 1):
        x = i * tile_width_cm * cm
        c.line(x, 0, x, height)

    # Draw horizontal lines
    for j in range(rows + 1):
        y = j * tile_height_cm * cm
        c.line(0, y, width, y)

    # Save the PDF
    c.save()
    print(
        f"✅ Saved A4 grid with {tile_width_cm}cm × {tile_height_cm}cm tiles as: {filename}"
    )


draw_tile_grid_on_a4()
