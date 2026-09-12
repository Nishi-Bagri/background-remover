from rembg import remove, new_session


# Local model for simple images.
simple_session = new_session("u2net")


def remove_background(image):
    """
    Remove the background using the local RMBG pipeline.

    This pipeline is used only for images classified
    as simple by the complexity analyzer.
    """

    output_image = remove(
        image,
        session=simple_session
    )

    return output_image