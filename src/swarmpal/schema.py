from __future__ import annotations

import datetime

import jsonschema
from yaml import safe_load

fetch_data_schema_txt = """
type: object
properties:
  data_params:
    type: array
    items:
      anyOf:
        - $ref: "#/definitions/vires_data_params"
        - $ref: "#/definitions/hapi_data_params"
    description: "List of data parameter objects, each conforming to either vires_data_params or hapi_data_params."
  process_params:
    type: array
    items:
      type: object
      properties:
        process_name:
          type: string
          description: "Name of the SwarmPAL process to run"
          enum:
            - FAC_single_sat
    description: "List of parameter objects, each describing a SwarmPal process that will be applied to the dataset"
required:
  - data_params
additionalProperties: false

definitions:
  common_data_params:
    pad_times:
      type: array
      prefixItems:
        - type: string
          format: timedelta
        - type: string
          format: timedelta
      description: "A tuple of time steps (in HH:MM:SS) to add before and after the timeseries"

  vires_data_params:
    type: object
    properties:

      provider:
        type: string
        description: "Data provider name."
        enum:
          - "vires"
      collection:
        type: string
        description: "Name of the data collection."
      measurements:
        type: array
        items:
          type: string
        description: "List of measurement fields to retrieve."
      models:
        type: array
        items:
          type: string
        description: "List of models to use."
      start_time:
        type: string
        format: iso8601-date-time
        description: "Start time for the data retrieval in ISO 8601 format."
      end_time:
        type: string
        format: iso8601-date-time
        description: "End time for the data retrieval in ISO 8601 format."
      server_url:
        type: string
        format: uri
        description: "URL of the server to connect to."
      pad_times:
        $ref: "#/definitions/common_data_params/pad_times"
      filters:
        type: array
        items:
          type: string
        description: "Logical filters to reduce the data volume"
      options:
        type: object
        properties:
          asynchronous:
            type: boolean
            description: "Submit request as an asynchronous job."
          show_progress:
            type: boolean
            description: "Whether to show progress during the operation."
        additionalProperties: true
        description: "Additional options for the data retrieval."
    required:
      - provider
      - collection
      - start_time
      - end_time
    additionalProperties: false
    description: "Schema for VirES data parameters configuration."

  hapi_data_params:
    type: object
    properties:
      provider:
        type: string
        description: "Data provider name."
        enum:
          - "hapi"
      dataset:
        type: string
        description: "Identifier of the HAPI dataset."
      parameters:
        type: string
        description: "Comma separated string of parameters to retrieve from the HAPI dataset."
      start:
        type: string
        format: iso8601-date-time
        description: "Start time for the data retrieval in ISO 8601 format."
      stop:
        type: string
        format: iso8601-date-time
        description: "End time for the data retrieval in ISO 8601 format."
      server:
        type: string
        format: uri
        description: "URL of the HAPI server to connect to."
      pad_times:
        $ref: "#/definitions/common_data_params/pad_times"
      options:
        type: object
        additionalProperties: true
        description: "Additional options for the HAPI data retrieval."
    required:
      - provider
      - dataset
      - start
      - stop
      - server
    additionalProperties: false
    description: "Schema for HAPI data parameters configuration."
"""

fetch_data_schema = safe_load(fetch_data_schema_txt)


def validate(config):
    """Validates config against the schema for fetch_data"""
    format_checker = jsonschema.FormatChecker()

    @format_checker.checks("iso8601-date-time")
    def is_iso8601_datetime(instance):
        try:
            datetime.datetime.fromisoformat(instance)
            return True
        except ValueError:
            return False

    @format_checker.checks("timedelta")
    def is_timedelta(instance):
        try:
            datetime.datetime.strptime(instance, "%H:%M:%S")
            return True
        except ValueError:
            return False

    BaseValidator = jsonschema.Draft202012Validator
    BaseValidator(schema=fetch_data_schema, format_checker=format_checker).validate(
        config
    )
