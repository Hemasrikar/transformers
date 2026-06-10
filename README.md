

## Installing Dependencies

The package manager used for this project is `uv`

`uv` package manager is recommended but the dependencies can also be installed using `pip` 
```bash
pip install .
```

### Dependencies Installation using uv

Install the `uv` pacakge manager first, using the following

For Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
For MacOs and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or Using pip package manager
```bash
pipx install uv
```
After installing run 
```bash
uv sync
```
to install all the required pacakges for this project

---

## Data Extraction

The branch `data` contains the jupyter notebooks related data extraction, data processing and validation.


> [!NOTE]
> Other branches contrains different versions of Architecture, they are experimental to find the best performing model.




