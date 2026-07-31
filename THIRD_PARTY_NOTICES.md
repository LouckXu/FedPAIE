# Third-party notices

This code release depends on third-party Python packages listed in
`requirements.txt`. Those packages retain their own licenses and copyrights.

The compact LUT architecture follows the formulation described by FedPAIE and
CLUT-Net:

- Fengyi Zhang, Hui Zeng, Tianjun Zhang, and Lin Zhang, *CLUT-Net: Learning
  Adaptively Compressed Representations of 3D Look-Up Tables for Lightweight Image
  Enhancement*, ACM MM 2022.
- Reference implementation: <https://github.com/Xian-Bei/CLUT>

The CLUT implementation in this repository was independently rewritten for the public
release using native PyTorch operations. No upstream source file, custom binary
extension, dataset, or pretrained weight is included. At the time this release
candidate was prepared, no license file was visible in the referenced CLUT repository.
Maintainers should obtain or confirm any permissions needed for their intended
distribution.

The Flickr-AES and MIT-Adobe FiveK datasets are not included. Their images, labels, and
derived artifacts remain subject to the terms published by their respective owners.
