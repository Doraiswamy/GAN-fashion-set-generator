from rest_framework import views, status
from rest_framework.response import Response

from .deployment.gan_deployment import generator_test


class GANGenerator(views.APIView):

    def get(self, request):
        image_response = generator_test()
        return Response(data=image_response, status=status.HTTP_200_OK)

