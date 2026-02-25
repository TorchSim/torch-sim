"""Package-level duecredit citations for TorchSim and its core dependencies.

This module must be imported at the end of torch_sim.__init__ so that all
packages are fully loaded before citations are registered.
"""

from torch_sim._duecredit import BibTeX, due


if due is not None:
    due.cite(
        BibTeX(
            """@article{cohen2025torchsim,
  title={TorchSim: An efficient atomistic simulation engine in PyTorch},
  author={Cohen, Orion and Riebesell, Janosh and Goodall, Rhys and
          Kolluru, Adeesh and Falletta, Stefano and Krause, Joseph and
          Colindres, Jorge and Ceder, Gerbrand and Gangan, Abhijeet S},
  journal={AI for Science},
  volume={1},
  number={2},
  pages={025003},
  year={2025},
  publisher={IOP Publishing},
  doi={10.1088/3050-287X/ae1799}
}"""
        ),
        description="TorchSim simulation engine",
        path="torch_sim",
        cite_module=True,
    )
    due.cite(
        BibTeX(
            """@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and
          Lerer, Adam and Bradbury, James and Chanan, Gregory and
          Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and
          Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}"""
        ),
        description="PyTorch deep learning framework",
        path="torch",
        cite_module=True,
    )
