📁 data/

**Drop your watershed data files here, then push to GitHub.**

---

## Option A — Individual files (recommended)

```
data/
├── dem.tif                  ← Filled SRTM 30m DEM (GeoTIFF)
├── subbasins.shp            ← 5 subbasins
├── subbasins.dbf
├── subbasins.shx
├── subbasins.prj
├── streams.shp              ← Stream network
├── streams.dbf
├── streams.shx
├── streams.prj
├── stream_order.shp         ← With Strahler order attribute
├── stream_order.dbf
├── stream_order.shx
├── stream_order.prj
├── flow_direction.tif       ← D8 flow direction
├── flow_accumulation.tif    ← D8 flow accumulation
└── pour_points.shp          ← Outlet pour points (+ sidecars)
```

## Option B — Single ZIP

Drop a single `watershed_data.zip` containing all the above.  
Section 0 auto-extracts and auto-detects all layers by filename keywords.

---

## Filename keywords recognised

| Layer | Recognised keywords |
|-------|---------------------|
| DEM | `dem`, `srtm`, `elevation`, `filled`, `fill` |
| Flow direction | `flowdir`, `flow_dir`, `fdir`, `direction` |
| Flow accumulation | `flowacc`, `flow_acc`, `facc`, `accumulation` |
| Subbasins | `subbasin`, `sub_basin`, `watershed`, `basin`, `catchment` |
| Streams | `stream`, `river`, `channel`, `network`, `drainage` |
| Stream order | same as streams but also contains `order` |
| Pour points | `pour`, `outlet`, `point` |

---

## After uploading

```bash
git add data/
git commit -m "feat: upload watershed data"
git push origin main
```

GitHub Actions fires within seconds.  
Check the **Actions** tab for progress — outputs appear in `outputs/` in ~5–10 minutes.

> **Large files (>100 MB)?**  
> Use [Git LFS](https://git-lfs.github.com):
> ```bash
> git lfs track "data/*.tif"
> git add .gitattributes
> git add data/dem.tif
> git commit -m "add DEM via LFS"
> ```

